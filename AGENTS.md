# AGENTS.md - Plasma Garden

WebGPU-based game that grows a procedural psychedelic garden using ECS architecture
with GPU-accelerated raymarched rendering.

## Build & Development Commands

```bash
npm run dev      # Development server with hot reload
npm run build    # Type-check (tsc) and build for production
```

**No tests exist.** The build command runs TypeScript type-checking before bundling.

Format code: `npx prettier --write "src/**/*.ts"`

## Tech Stack

- **TypeScript 5.9** - Strict mode, ES2022 target
- **WebGPU** - GPU rendering and compute shaders
- **TypeGPU 0.9** - Type-safe WebGPU wrapper and shader compilation
- **BitECS 0.4** - Entity Component System framework
- **Vite** (rolldown-vite) - Build tool

## Project Structure

```
src/
  main.ts              # Entry point, game loop, world setup
  components/
    game/              # Rederable non-organism objects
    general/           # General components used on on various entities
    organisms/         # Procedural organisms
  lib/                 # Shared utilities (SDF, lighting, math, buffers, etc.)
```

## Code Style Guidelines

### Formatting (Prettier-enforced)

- No semicolons
- Single quotes
- Sorted imports with blank lines between groups (external first, then relative)

### TypeScript

- Strict mode enabled - all code must be type-safe
- No unused variables or parameters (build will fail)
- Use `type` imports: `import type { Foo } from './foo'`

### Naming Conventions

- **Components**: PascalCase noun (`Position`, `Velocity`, `GridPosition`)
- **Systems**: camelCase + `System` suffix (`physicsSystem`, `spawnKelpSystem`)
- **Render systems**: `createRender*System` (`createRenderKelpSystem`)
- **Entity creators**: `create*` prefix (`createPlayerEntity`)
- **Constants**: SCREAMING_SNAKE_CASE (`SPAWN_RATE`, `BOUNDARY`)

### ECS Architecture

Components are arrays indexed by entity ID. Define type and array with same name:

```ts
type MyComponent = { value: number; position: v2f }
const MyComponent = [] as MyComponent[]

// Usage
MyComponent[entityId] = { value: 1, position: vec2f(0, 0) }
```

Systems query and operate on entities:

```ts
export function mySystem(world: World) {
  for (const eid of query(world, [ComponentA, ComponentB])) {
    // Process entity
  }
}
```

### TypeGPU and Shaders

- Add `'use gpu'` directive to functions that may run on GPU
- Use TypeGPU vectors from `typegpu/data`: `vec2f()`, `vec3f()`, `vec4f()`
- Use TypeGPU math from `typegpu/std`: `length()`, `normalize()`, `dot()`, etc.
- Vector math via methods: `position.add(velocity.mul(delta))`
- Normal operators only for scalars: `position.x + velocity.y`

```ts
function sdSphere(p: v3f, radius: number): number {
  'use gpu'
  return length(p) - radius
}
```

### GPU Struct Definitions

```ts
import { struct, f32, vec3f, type Infer } from 'typegpu/data'

export const MyStruct = struct({ position: vec3f, scale: f32 })
export type MyStruct = Infer<typeof MyStruct>
```

### Adding New Features

1. Define components in `components/` subdirectory
2. Create systems as pure functions with component queries
3. Wire into `main.ts` `tick()` function
4. For GPU rendering: use `createSDFInstancesRenderer` from `lib/sdf-instance-pipeline.ts`
