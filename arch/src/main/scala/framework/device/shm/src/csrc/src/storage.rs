use std::collections::BTreeMap;

struct Node {
    data: Vec<u8>,
    initialized: Vec<bool>,
    descriptor: [u64; 4],
    fields: u8,
}

/// Device communication state, independent of core-local banks and thread execution.
pub struct Storage {
    nodes: BTreeMap<usize, Node>,
    links: Vec<(usize, usize)>,
    capacity: usize,
}

impl Storage {
    pub fn new(cores: &[usize], capacity: usize, links: Vec<(usize, usize)>) -> Self {
        assert!(capacity >= 16 && capacity.is_power_of_two());
        assert!(links
            .iter()
            .all(|(a, b)| cores.contains(a) && cores.contains(b) && a != b));
        let nodes = cores
            .iter()
            .map(|&core| {
                (
                    core,
                    Node {
                        data: vec![0; capacity],
                        initialized: vec![false; capacity],
                        descriptor: [0; 4],
                        fields: 0,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(nodes.len(), cores.len());
        Self { nodes, links, capacity }
    }

    pub fn request(&mut self, core: usize, operation: u32, address: u64, value: u64) -> u64 {
        assert!(self.nodes.contains_key(&core));
        match operation {
            13 => {
                assert!(address < 4);
                let node = self.nodes.get_mut(&core).unwrap();
                node.descriptor[address as usize] = value;
                node.fields |= 1 << address;
                0
            }
            14 => {
                assert_eq!(address, 0);
                let source = self.nodes.get_mut(&core).unwrap();
                assert_eq!(source.fields, 15, "Incomplete device transfer descriptor");
                source.fields = 0;
                let [from, target, to, bytes] = source.descriptor;
                assert!(bytes > 0 && bytes <= u32::MAX as u64);
                assert!(from % 16 == 0 && to % 16 == 0);
                if !self.links.contains(&(core, target as usize)) {
                    return 1;
                }
                assert!(from.checked_add(bytes).unwrap() <= self.capacity as u64);
                assert!(to.checked_add(bytes).unwrap() <= self.capacity as u64);
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
                0
            }
            15 => match address {
                0 => self.capacity as u64,
                1 => core as u64,
                2 => self.nodes.len() as u64,
                3 => u64::from(self.links.contains(&(core, value as usize))),
                _ => panic!("Invalid device query"),
            },
            16 | 17 => {
                assert!(address % 8 == 0 && address.checked_add(8).unwrap() <= self.capacity as u64);
                let node = self.nodes.get_mut(&core).unwrap();
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
        }
    }
}
