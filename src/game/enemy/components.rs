use crate::Vec2;
use bevy::prelude::Component;

#[derive(Component)]
pub struct Enemy {
    pub direction: Vec2,
}

#[derive(Component)]
pub struct Boss {
    pub speed_modifer: f32,
}
