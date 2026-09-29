// Optional enhancement: update the prediction instantly as sliders move.
// Without JavaScript the form still works through the "Predict" button.
(function () {
  const form = document.getElementById("predict-form");
  if (!form || !window.fetch) return;
  document.documentElement.classList.add("js");

  const sliders = form.querySelectorAll('input[type="range"]');
  const regimeEl = document.getElementById("result-regime");
  const confEl = document.getElementById("result-confidence");
  const colors = {};
  document.querySelectorAll(".bar-fill").forEach((el) => {
    colors[el.dataset.regime] = getComputedStyle(el).getPropertyValue("--c").trim();
  });

  let timer = null;
  let latest = 0;

  function showValue(input) {
    const out = document.getElementById("out-" + input.name);
    if (out) out.textContent = Number(input.value).toFixed(Number(input.dataset.digits));
  }

  async function predict() {
    const body = {};
    sliders.forEach((s) => { body[s.name] = Number(s.value) / Number(s.dataset.scale); });
    const id = ++latest;
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok || id !== latest) return;
      const data = await res.json();
      regimeEl.textContent = data.regime;
      regimeEl.style.setProperty("--c", colors[data.regime] || "");
      confEl.textContent = Math.round(data.probabilities[data.regime] * 100) + "% confidence";
      Object.entries(data.probabilities).forEach(([regime, p]) => {
        const fill = document.querySelector('.bar-fill[data-regime="' + regime + '"]');
        const val = document.querySelector('.bar-value[data-regime="' + regime + '"]');
        if (fill) fill.style.width = (p * 100).toFixed(1) + "%";
        if (val) val.textContent = Math.round(p * 100) + "%";
      });
    } catch (e) {
      // Network hiccup: the Predict button path still works, so show it again.
      document.documentElement.classList.remove("js");
    }
  }

  sliders.forEach((s) => {
    s.addEventListener("input", () => {
      showValue(s);
      clearTimeout(timer);
      timer = setTimeout(predict, 120);
    });
  });
  form.addEventListener("submit", (e) => { e.preventDefault(); predict(); });
})();
