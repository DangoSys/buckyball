#[derive(Default)]
pub(crate) struct ArbiterModel {
    selected: Option<u32>,
}

impl ArbiterModel {
    pub(crate) fn step(&mut self, valid: u32, ready: bool, last: u32) -> Option<u32> {
        let selected = self
            .selected
            .or_else(|| (valid != 0).then(|| valid.trailing_zeros()));
        let selected = selected?;
        if valid & (1 << selected) == 0 {
            return None;
        }

        self.selected = if ready && last & (1 << selected) != 0 {
            None
        } else {
            Some(selected)
        };
        Some(selected)
    }
}
