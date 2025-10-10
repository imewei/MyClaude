import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        '__tests__/**',
        '*.config.js',
        'agent-cli.mjs'  // CLI wrapper, tested via E2E
      ],
      lines: 80,
      functions: 80,
      branches: 75,
      statements: 80
    },
    include: ['__tests__/**/*.test.mjs'],
    testTimeout: 10000
  }
});
