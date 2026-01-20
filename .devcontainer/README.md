# Dev Container for Plasma Garden

This directory contains the development container configuration for the Plasma Garden WebGPU project.

## What's Included

- **Node.js 20** with JavaScript/TypeScript support
- **OpenCode CLI** for development assistance
- **Port forwarding** for Vite dev server (5173)
- **Automatic dependency installation** on container creation

## Usage

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
2. Open this project in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. The container will be built and dependencies installed automatically

## Development

Once in the container:

- Run `npm run dev` to start the Vite development server
- The server will be available at `http://localhost:5173`
- TypeScript and Prettier are pre-configured

## Container Features

- Node.js 20 runtime
- TypeScript 5.9.3
- Git and GitHub CLI
- OpenCode CLI for development assistance
- Automatic port forwarding for development server
