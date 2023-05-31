pub struct GameOver(pub bool);
impl Default for GameOver {
    fn default() -> Self {
        GameOver(false)
    }
}
