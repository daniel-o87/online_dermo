/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}, // Ensure autoprefixer is included if you need it
  },
};

export default config;
