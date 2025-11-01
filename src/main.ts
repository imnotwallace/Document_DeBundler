console.log("main.ts is executing!");

import "./app.css";
import App from "./App.svelte";

console.log("About to mount App...");
console.log("Target element:", document.getElementById("app"));

const app = new App({
  target: document.getElementById("app")!,
});

console.log("App mounted successfully!");

export default app;
