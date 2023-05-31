use bevy::{prelude::Component, time::Timer};

#[derive(Component)]
pub struct Collectable;

#[derive(Component)]
pub struct ConcreteModeCollectable;


#[derive(Component)]
pub struct ConcreteModeTimer {
    pub timer: Timer,
    pub almost_elapsed: bool
}

#[derive(PartialEq)]
pub enum PlayerSkin {
    Concrete,
    Usual
}

#[derive(Component)]
pub struct ConcreteModeElapsingTimer {
    pub timer: Timer,
    pub player_skin: PlayerSkin,
}

