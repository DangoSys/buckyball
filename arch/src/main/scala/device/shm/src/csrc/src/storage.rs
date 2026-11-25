use std::collections::BTreeMap;

struct Node {
    data: Vec<u8>,
    initialized: Vec<bool>,
    descriptor: [u64; 4],
    fields: u8,
}

/// Device communication state, independent of hart-local banks and thread execution.
pub struct Storage {
    nodes: BTreeMap<usize, Node>,
    links: Vec<(usize, usize)>,
    capacity: usize,
}

impl Storage {
    /// `links` are the kept links of the device SHM network (framework/device/shm/Network.scala): a tree over the
    /// harts, a write routed along its one path between two harts.
    pub fn new(harts: &[usize], capacity: usize, links: Vec<(usize, usize)>) -> Self {
        assert!(capacity >= 16 && capacity.is_power_of_two());
        assert!(links
            .iter()
            .all(|(a, b)| harts.contains(a) && harts.contains(b) && a != b));
        assert_eq!(
            links.len() + 1,
            harts.len(),
            "Device links must form a tree over the harts"
        );
        let nodes = harts
            .iter()
            .map(|&hart| {
                (
                    hart,
                    Node {
                        data: vec![0; capacity],
                        initialized: vec![false; capacity],
                        descriptor: [0; 4],
                        fields: 0,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(nodes.len(), harts.len());
        let storage = Self {
            nodes,
            links,
            capacity,
        };
        for &from in harts {
            for &to in harts {
                if from != to {
                    storage.route(from, to);
                }
            }
        }
        storage
    }

    /// The harts on the path of the tree from `from` to `to`, both ends included.
    fn route(&self, from: usize, to: usize) -> Vec<usize> {
        let mut paths = vec![vec![from]];
        loop {
            let path = paths.pop().expect("Device links do not connect the harts");
            let last = *path.last().unwrap();
            if last == to {
                return path;
            }
            for &(a, b) in &self.links {
                let next = if a == last {
                    b
                } else if b == last {
                    a
                } else {
                    continue;
                };
                if !path.contains(&next) {
                    paths.push([path.as_slice(), &[next]].concat());
                }
            }
        }
    }

    /// The value of a request and the cycles it holds the requesting hart.
    pub fn request(&mut self, hart: usize, operation: u32, address: u64, value: u64) -> (u64, u64) {
        assert!(self.nodes.contains_key(&hart));
        let value = match operation {
            13 => {
                assert!(address < 4);
                let node = self.nodes.get_mut(&hart).unwrap();
                node.descriptor[address as usize] = value;
                node.fields |= 1 << address;
                0
            }
            14 => {
                assert_eq!(address, 0);
                let node = self.nodes.get_mut(&hart).unwrap();
                assert_eq!(node.fields, 15, "Incomplete device transfer descriptor");
                node.fields = 0;
                let [from, target, to, bytes] = node.descriptor;
                assert!(bytes > 0 && bytes <= u32::MAX as u64);
                assert!(from % 16 == 0 && to % 16 == 0);
                if target as usize == hart || !self.nodes.contains_key(&(target as usize)) {
                    return (1, 0);
                }
                // The routers lock the path for each burst, which stays within a 4 KiB page at both ends; the
                // start returns after the B of the last one.
                let routers = self.route(hart, target as usize).len() as u64;
                // Uncontended Control request-to-reply timing. AW locks one router per cycle,
                // overlapping the four-cycle source-to-W pipeline; B returns combinationally.
                let route_setup = routers.saturating_sub(3);
                let mut cycles = 2;
                let mut offset = 0;
                while offset < bytes {
                    let burst = (bytes - offset)
                        .min(4096 - (from + offset) % 4096)
                        .min(4096 - (to + offset) % 4096);
                    let beats = burst.div_ceil(16);
                    // Four source credits issue four beats in five cycles. The destination
                    // write queue, SRAM replies and B/plan states add nine cycles per burst;
                    // the command's initial/final Control handshakes add the two above.
                    cycles += 9 + route_setup + beats + beats.div_ceil(4);
                    offset += burst;
                }
                assert!(from.checked_add(bytes).unwrap() <= self.capacity as u64);
                assert!(to.checked_add(bytes).unwrap() <= self.capacity as u64);
                let source = &self.nodes[&hart];
                let from = from as usize..(from + bytes) as usize;
                assert!(
                    source.initialized[from.clone()].iter().all(|&value| value),
                    "Device transfer reads unpublished bytes"
                );
                let data = source.data[from].to_vec();
                let target = self.nodes.get_mut(&(target as usize)).unwrap();
                let to = to as usize..(to + bytes) as usize;
                target.data[to.clone()].copy_from_slice(&data);
                target.initialized[to].fill(true);
                return (0, cycles);
            }
            15 => match address {
                0 => self.capacity as u64,
                1 => hart as u64,
                2 => self.nodes.len() as u64,
                3 => {
                    u64::from(value as usize != hart && self.nodes.contains_key(&(value as usize)))
                }
                _ => panic!("Invalid device query"),
            },
            16 | 17 => {
                assert!(
                    address % 8 == 0 && address.checked_add(8).unwrap() <= self.capacity as u64
                );
                let node = self.nodes.get_mut(&hart).unwrap();
                let range = address as usize..address as usize + 8;
                if operation == 17 {
                    node.data[range.clone()].copy_from_slice(&value.to_le_bytes());
                    node.initialized[range].fill(true);
                    0
                } else {
                    assert!(
                        node.initialized[range.clone()].iter().all(|&value| value),
                        "Device read of unpublished bytes"
                    );
                    u64::from_le_bytes(node.data[range].try_into().unwrap())
                }
            }
            _ => panic!("Invalid device management operation"),
        };
        (value, 0)
    }
}
