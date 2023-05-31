pub struct AddCollectableEvent {
    pub amount: usize,
}
impl Default for AddCollectableEvent {
    fn default() -> Self {
        AddCollectableEvent { amount: 0 }
    }
}

pub struct SetConcreteModeActive;

impl Default for SetConcreteModeActive {
    fn default() -> Self {
        SetConcreteModeActive
    }
}
