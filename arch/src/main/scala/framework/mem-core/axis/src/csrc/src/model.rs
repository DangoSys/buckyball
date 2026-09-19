use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Beat {
    pub(crate) data: u32,
    pub(crate) keep: u8,
    pub(crate) last: bool,
}

#[derive(Default)]
pub(crate) struct AxiSModel {
    beats: VecDeque<Beat>,
}

impl AxiSModel {
    pub(crate) fn push(&mut self, beat: Beat) {
        self.beats.push_back(beat);
    }

    pub(crate) fn pop(&mut self) -> Option<Beat> {
        self.beats.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_beat_order_and_packet_boundaries() {
        let mut model = AxiSModel::default();
        let first = Beat {
            data: 0x1234_5678,
            keep: 0xf,
            last: false,
        };
        let second = Beat {
            data: 0x9abc_def0,
            keep: 0x3,
            last: true,
        };

        model.push(first);
        model.push(second);

        assert_eq!(model.pop(), Some(first));
        assert_eq!(model.pop(), Some(second));
        assert_eq!(model.pop(), None);
    }
}
