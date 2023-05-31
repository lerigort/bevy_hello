use bevy::{prelude::Resource, time::Timer};

#[derive(Resource)]
pub struct ChangeDirectionTimer {
    pub timer: Timer,
    pub sound_not_played: bool,
}
