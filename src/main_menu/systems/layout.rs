use bevy::prelude::*;
use bevy::text::BreakLineOn::WordBoundary;

use crate::main_menu::{
    components::{MainMenu, PlayButton, QuitButton},
    styles::{BUTTON_STYLE, MENU_STYLE, NORMAL_BUTTON_COLOR},
};

pub fn build_main_menu(commands: &mut Commands, asset_server: &Res<AssetServer>) -> Entity {
    let main_menu_entity = commands
        .spawn((
            // Main Menu
            NodeBundle {
                style: MENU_STYLE,
                ..default()
            },
            MainMenu {},
        ))
        .with_children(|parent| {
            // Title
            parent
                .spawn(NodeBundle {
                    style: Style {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        size: Size {
                            width: Val::Px(300.0),
                            height: Val::Px(120.0),
                        },
                        ..default()
                    },
                    ..default()
                })
                .with_children(|parent| {
                    // Image 1
                    parent.spawn(ImageBundle {
                        style: Style {
                            size: Size::new(Val::Px(32.0), Val::Px(32.0)),
                            margin: UiRect {
                                left: (Val::Px(8.0)),
                                right: (Val::Px(8.0)),
                                top: (Val::Px(8.0)),
                                bottom: (Val::Px(8.0)),
                            },
                            ..default()
                        },
                        image: asset_server
                            .load("kenney_pixel-platformer/Characters/character_0000_l.png")
                            .into(),
                        ..default()
                    });
                })
                .with_children(|parent| {
                    parent.spawn(TextBundle {
                        text: Text {
                            sections: vec![TextSection::new(
                                "The game",
                                TextStyle {
                                    font: asset_server.load("fonts/Pangolin-Regular.ttf"),
                                    color: Color::SEA_GREEN,
                                    font_size: 85.0,
                                },
                            )],
                            linebreak_behaviour: WordBoundary,
                            alignment: TextAlignment::Center,
                        },
                        ..default()
                    });
                })
                .with_children(|parent| {
                    // Image 2
                    parent.spawn(ImageBundle {
                        style: Style {
                            size: Size::new(Val::Px(32.0), Val::Px(32.0)),
                            margin: UiRect {
                                left: (Val::Px(8.0)),
                                right: (Val::Px(8.0)),
                                top: (Val::Px(8.0)),
                                bottom: (Val::Px(8.0)),
                            },
                            ..default()
                        },
                        image: asset_server
                            .load("kenney_pixel-platformer/Characters/character_0020.png")
                            .into(),
                        ..default()
                    });
                });
        })
        .with_children(|parent| {
            // Button 1
            parent
                .spawn((
                    ButtonBundle {
                        style: BUTTON_STYLE,
                        background_color: NORMAL_BUTTON_COLOR.into(),
                        ..default()
                    },
                    PlayButton {},
                ))
                .with_children(|parent| {
                    parent.spawn(TextBundle {
                        text: Text {
                            sections: vec![TextSection::new(
                                "Play!",
                                TextStyle {
                                    font: asset_server.load("fonts/Pangolin-Regular.ttf"),
                                    color: Color::WHITE,
                                    font_size: 32.0,
                                },
                            )],
                            linebreak_behaviour: WordBoundary,
                            alignment: TextAlignment::Center,
                        },
                        ..default()
                    });
                });
        })
        .with_children(|parent| {
            // Button 2
            parent
                .spawn((
                    ButtonBundle {
                        style: BUTTON_STYLE,
                        background_color: NORMAL_BUTTON_COLOR.into(),
                        ..default()
                    },
                    QuitButton {},
                ))
                .with_children(|parent| {
                    parent.spawn(TextBundle {
                        text: Text {
                            sections: vec![TextSection::new(
                                "Quit!",
                                TextStyle {
                                    font: asset_server.load("fonts/Pangolin-Regular.ttf"),
                                    color: Color::WHITE,
                                    font_size: 32.0,
                                },
                            )],
                            linebreak_behaviour: WordBoundary,
                            alignment: TextAlignment::Center,
                        },
                        ..default()
                    });
                });
        })
        .id();

    main_menu_entity
}

pub fn spawn_menu(mut commands: Commands, asset_server: Res<AssetServer>) {
    let _main_menu = build_main_menu(&mut commands, &asset_server);
}

pub fn despawn_menu(mut commands: Commands, menu_query: Query<Entity, With<MainMenu>>) {
    if let Ok(menu) = menu_query.get_single() {
        commands.entity(menu).despawn_recursive();
    }
}
