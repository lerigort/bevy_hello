mod collectables;
mod enemy;
mod mechanics;
pub mod player;
mod systems;

use bevy::prelude::*;
use collectables::systems::CollectablesPlugin;
use enemy::systems::EnemyPlugin;
use mechanics::systems::MechanicsPlugin;
use player::systems::PlayerPlugin;

use crate::AppState;

use self::systems::toogle_simulation;

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_state::<SimulationState>()
            .add_plugin(PlayerPlugin)
            .add_plugin(EnemyPlugin)
            .add_plugin(CollectablesPlugin)
            .add_plugin(MechanicsPlugin)
            .add_system(toogle_simulation.run_if(in_state(AppState::Game)));
    }
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimulationState {
    Running,
    #[default]
    Paused,
}
