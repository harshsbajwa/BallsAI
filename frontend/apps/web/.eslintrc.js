/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  extends: ["@nba/eslint-config", "next/core-web-vitals"],
  parserOptions: {
    project: "./tsconfig.json",
  },
};