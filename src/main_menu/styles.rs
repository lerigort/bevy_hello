use bevy::prelude::*;

pub const NORMAL_BUTTON_COLOR: Color = Color::SILVER;
pub const PRESSED_BUTTON_COLOR: Color = Color::TURQUOISE;
pub const HOVERED_BUTTON_COLOR: Color = Color::GRAY;

pub const BUTTON_STYLE: Style = Style {
    size: Size::new(Val::Px(160.0), Val::Px(80.0)),
    justify_content: JustifyContent::Center,
    align_content: AlignContent::Center,
    ..Style::DEFAULT
};

pub const MENU_STYLE: Style = Style {
    flex_direction: FlexDirection::Column,
    justify_content: JustifyContent::Center,
    size: Size::new(Val::Percent(100.0), Val::Percent(100.0)),
    align_items: AlignItems::Center,
    gap: Size::new(Val::Px(8.0), Val::Px(8.0)),
    ..Style::DEFAULT
};
