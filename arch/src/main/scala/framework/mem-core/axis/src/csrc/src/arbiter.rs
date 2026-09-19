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

        if ready {
            if last & (1 << selected) != 0 {
                self.selected = None;
            } else {
                self.selected = Some(selected);
            }
        }
        Some(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_until_last_then_rearbitrates() {
        let mut model = ArbiterModel::default();

        assert_eq!(model.step(0b11, true, 0b00), Some(0));
        assert_eq!(model.step(0b11, true, 0b10), Some(0));
        assert_eq!(model.step(0b11, true, 0b01), Some(0));
        assert_eq!(model.step(0b10, true, 0b10), Some(1));
        assert_eq!(model.step(0b00, true, 0b00), None);
    }

    #[test]
    fn preserves_selection_during_backpressure() {
        let mut model = ArbiterModel::default();

        assert_eq!(model.step(0b10, false, 0b00), Some(1));
        assert_eq!(model.step(0b11, true, 0b01), Some(0));
    }
}
