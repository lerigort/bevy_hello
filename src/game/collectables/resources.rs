use bevy::{prelude::Resource, time::Timer};

#[derive(Resource)]
pub struct Score {
    pub score: usize,
}

// with every collected point chance to get ConcreteMode rise
// thought, we want to drop chance back to minimum, after collecting ConcreteMode
#[derive(Resource)]
pub struct ConcreteChance {
    pub chance: f32,
}

#[derive(Resource)]
pub struct IsInConcreteMode {
    pub status: bool,
}
