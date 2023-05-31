use std::sync::Mutex;

use bevy::prelude::Vec2;

use crate::game::collectables::resources::Score;

use self::{components::Enemy, events::AddEnemyEvent, resources::ChangeDirectionTimer};

pub const ENEMY_SPEED: f32 = 100.0;
pub const ENEMY_SIZE: f32 = 12.0;
pub const ENEMY_CHANGE_DIRECTION_CYCLE: u64 = 5;
pub const NUMBER_OF_ENEMIES: usize = 6;

pub const BOSS_DIRECTION: bevy::prelude::Vec2 = Vec2::new(0.1, 0.9);
pub const BOSS_SIZE: f32 = 12.0;
pub const BOSS_SPEED: f32 = 550.0;

pub mod components;
pub mod events;
pub mod resources;
pub mod systems;
