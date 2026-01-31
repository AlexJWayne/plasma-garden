import { createRenderBackgroundSystem } from '../components/game/background'
import { createRenderPlayerSystem } from '../components/game/player'
import { createRenderKelpSystem } from '../components/organisms/kelp'
import { createRenderMushroomSystem } from '../components/organisms/mushroom'
import { clearScreenSystem } from '../lib/clear-screen'
import type { World } from '../main'

export function createRenderWorldSystem(world: World) {
  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderMushroomSystem = createRenderMushroomSystem(world)
  const renderKelpSystem = createRenderKelpSystem(world)

  return function renderWorldSystem() {
    clearScreenSystem(world)
    renderBackgroundSystem()
    renderPlayerSystem()
    renderMushroomSystem()
    renderKelpSystem()
  }
}
