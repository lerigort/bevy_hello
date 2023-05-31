use super::player::{components::Player, PLAYER_SIZE};

pub const INITIAL_AMOUNT_OF_COLLECTABLES: u64 = 10;
pub const COLLECTABLE_SIZE: f32 = 15.0;

pub const INITIAL_CHANCE_OF_CONCRETE_MODE: f32 = 0.01;
pub const RISE_STEP_OF_CHANCE_OF_CONCRETE_MODE: f32 = 0.01;
pub const CONCRETE_MODE_DURATION: u64 = 5;

pub mod components;
pub mod events;
pub mod resources;
pub mod systems;
