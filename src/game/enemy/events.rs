pub struct AddEnemyEvent {
    pub amount: usize,
}
impl Default for AddEnemyEvent {
    fn default() -> Self {
        AddEnemyEvent { amount: 0 }
    }
}

pub struct AddBossEvent;
impl Default for AddBossEvent {
    fn default() -> Self {
        AddBossEvent {}
    }
}

pub struct SpeedUpBoss;
impl Default for SpeedUpBoss {
    fn default() -> Self {
        SpeedUpBoss {}
    }
}
