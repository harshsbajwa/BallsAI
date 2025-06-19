/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  extends: ["@nba/eslint-config"],
  parserOptions: {
    project: "./tsconfig.json",
  },
};