This is a WebGPU-based game/simulation built with TypeScript, using an Entity Component System (ECS) architecture via BitECS and GPU compute shaders via TypeGPU.

## Project Overview

**Project Name**: Plasma Garden
**Type**: Real-time WebGPU game/simulation
**Architecture**: Entity Component System (ECS) with GPU-accelerated rendering
**Target**: Modern browsers with WebGPU support

## Core Technologies

### Primary Stack
- **TypeScript 5.9.3** - Main language with strict type checking
- **WebGPU** - GPU rendering and compute
- **TypeGPU 0.8.2** - Type-safe WebGPU wrapper and shader compilation
- **BitECS 0.4.0** - Entity Component System framework
- **Vite** - Build tool

### Supporting Libraries
- **wgpu-matrix 3.4.0** - Matrix math for 3D/2D transformations
- **@typegpu/sdf 0.8.0** - Signed Distance Field utilities
- **Prettier** - Code formatting with import sorting

### Key Structure Notes
- **`/game`**: Contains ECS components, game systems, and entity logic
- **`/lib`**: Shared utilities for GPU operations, math, geometry, and helpers
- **Root files**: Core initialization, WebGPU setup, and main entry point

## Code Style and Patterns

### TypeScript Configuration
- **Target**: ES2022 with modern features
- **Strict mode**: Enabled with additional linting rules
- **Module system**: ESNext with bundler resolution
- **Unused variables/parameters**: Not allowed

### Coding Standards
- **Semicolons**: Disabled (Prettier config)
- **Quotes**: Single quotes preferred
- **Import organization**: Sorted with relative imports last
- **Import separation**: Blank lines between import groups
- All of the above is enforced by Prettier.

### ECS Architecture Patterns
- **Components**: Pure data arrays (no methods)
- **Systems**: Functions that operate on component arrays
- **World**: Central state container with all systems and resources
- **Entities**: Numeric IDs managed by BitECS

### TypeGPU

- **Vectors**: Always uses vector creation functions from `typegpu/data`.
- **Math**: Always uses math functions from `typegpu/std` where possible to let those functions potentially be used in shaders.
- **Use GPU directives**: Uses `'use gpu'` directives in any pure math function that might be run on the GPU.

## Development Guidelines

### When Adding New Features
1. **Components first**: Define data structures in `components.ts`
2. **Systems second**: Create processing logic as pure functions
3. **Integration**: Wire systems into the main game loop
4. **GPU resources**: Use TypeGPU for any shader-based rendering
