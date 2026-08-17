// eslint flat config（ESLint 10 + typescript-eslint 8）。
//
// 只覆盖 TS 源码与测试文件；dist/node_modules 排除。类型感知规则（no-floating-promises 等）
// 需要 parserOptions.projectService 指向本目录的 tsconfig，故 tsconfig.json 的 include 必须
// 同时覆盖 src 与 tests（见 tsconfig.json 注释），否则测试文件会被类型感知规则跳过或报错。
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    // eslint.config.js 自身是 .js（非 tsconfig 覆盖范围），typescript-eslint 的 base
    // 配置会给所有文件挂载类型感知 parser——不排除会导致对本文件本身报
    // "not found by the project service"。同理排除 tests/fixtures/**/*.mjs：那是
    // 纯 JS 的假 stdio MCP 后端 fixture（node 直接跑，不编译、不进 tsconfig），
    // 不该套用 TS 侧的类型感知规则。
    ignores: ["dist/**", "node_modules/**", "eslint.config.js", "tests/fixtures/**/*.mjs"],
  },
  tseslint.configs.eslintRecommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    // 只对 tsconfig.json include 覆盖到的 .ts 开类型感知规则——不加 files 限制会把
    // 这块 languageOptions 应用到本文件（eslint.config.js 自身）等非 tsconfig 覆盖
    // 范围的文件，触发 projectService "not found by the project service" 报错。
    files: ["src/**/*.ts", "tests/**/*.ts", "vitest.config.ts"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // 红线：跨语言/协议边界代码不允许 any 兜底，未知类型走 unknown + 收窄。
      "@typescript-eslint/no-explicit-any": "error",
      // 红线：I/O 全 async/await，不吞 Promise（无 floating promise）。
      "@typescript-eslint/no-floating-promises": "error",
    },
  },
);
