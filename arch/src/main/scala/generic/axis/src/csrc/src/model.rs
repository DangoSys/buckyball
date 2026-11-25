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
