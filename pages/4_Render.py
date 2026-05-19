import json
import colorsys

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from skimage import color as skcolor

st.set_page_config(page_title="Render", layout="wide")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"

# ── Species selector ───────────────────────────────────────────────────────────
species_list = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
if not species_list:
    st.warning("No species found. Add one on the Home page.")
    st.stop()

active = st.session_state.get("active_species", species_list[0])
if active not in species_list:
    active = species_list[0]

with st.sidebar:
    st.header("Render")
    active = st.selectbox("Species", species_list,
                          index=species_list.index(active) if active in species_list else 0)
    st.session_state.active_species = active

SPECIES_DIR = DATA_DIR / active
PARTS_PATH  = SPECIES_DIR / "parts.json"
CSV_PATH    = SPECIES_DIR / "output" / "color_data.csv"

st.title(f"Iridescence Renderer: {active}")
st.caption("Each sphere is driven by the measured color data for that body part. "
           "Drag to spin it. The orbiting light reveals how hue shifts with viewing angle, "
           "which is the defining signature of structural color.")

# ── Load parts ─────────────────────────────────────────────────────────────────
if not PARTS_PATH.exists():
    st.warning("No parts.json found. Complete the Annotate step first.")
    st.stop()

with open(PARTS_PATH) as f:
    parts_meta = json.load(f)

# ── Load color data (optional) ─────────────────────────────────────────────────
df = None
if CSV_PATH.exists():
    df = pd.read_csv(str(CSV_PATH), index_col="frame")

# ── Build per-part shader parameters ──────────────────────────────────────────
def _hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    h_val, l_val, s_val = colorsys.rgb_to_hls(r, g, b)
    return h_val * 360.0, s_val, l_val

def _lab_to_hsl(L: float, A: float, B: float) -> tuple[float, float, float]:
    """Convert CIE LAB values to proper HSL for the shader.

    CIELAB hue angles and HSL hue angles are completely different coordinate
    systems — passing a raw LAB hue to an HSL shader gives wrong colours.
    This function goes LAB → sRGB → HSL to get the correct mapping.
    """
    lab = np.array([[[L, A, B]]], dtype=float)
    rgb = skcolor.lab2rgb(lab)                        # → 0-1 linear sRGB
    r, g, b = (float(np.clip(rgb[0, 0, c], 0.0, 1.0)) for c in range(3))
    h_hls, l_hls, s_hls = colorsys.rgb_to_hls(r, g, b)
    # Boost lightness so dark-but-vivid feathers don't render as near-black
    l_shader = float(np.clip(l_hls + 0.18, 0.25, 0.78))
    s_shader = float(np.clip(s_hls,         0.30, 1.00))
    return h_hls * 360.0, s_shader, l_shader

def _ii_level(ii: float) -> str:
    if ii > 8:  return "Strong"
    if ii > 4:  return "Moderate"
    if ii > 1:  return "Low-moderate"
    return "Low"

parts_data = []
for p in parts_meta:
    name  = p["name"]
    color = p["color"]
    base_hue, base_sat, base_lit = _hex_to_hsl(color)

    if df is not None and f"{name}_hue" in df.columns:
        hue_s    = df[f"{name}_hue"].dropna()
        chroma_s = df[f"{name}_chroma"].dropna()
        L_col    = f"{name}_L_norm" if f"{name}_L_norm" in df.columns else f"{name}_L"
        L_s      = df[L_col].dropna() if L_col in df.columns else pd.Series(dtype=float)
        A_s      = df[f"{name}_A"].dropna() if f"{name}_A" in df.columns else pd.Series(dtype=float)
        B_s      = df[f"{name}_B"].dropna() if f"{name}_B" in df.columns else pd.Series(dtype=float)

        if len(hue_s) >= 5:
            # Hue spread — circular std dev in CIELAB (valid angular quantity)
            rad = np.radians(hue_s.values)
            sin_m, cos_m = np.sin(rad).mean(), np.cos(rad).mean()
            R          = float(np.clip(np.sqrt(sin_m**2 + cos_m**2), 1e-9, 1.0))
            hue_spread = float(np.degrees(np.sqrt(-2.0 * np.log(R))))

            chroma_mean = float(chroma_s.mean()) if len(chroma_s) > 0 else 20.0
            ii = round(hue_spread * chroma_mean / 100.0, 3)

            # Base colour: convert mean LAB → proper HSL for the shader
            L_mean = float(L_s.mean()) if len(L_s) > 0 else 50.0
            A_mean = float(A_s.mean()) if len(A_s) > 0 else 0.0
            B_mean = float(B_s.mean()) if len(B_s) > 0 else 0.0
            hue_mean, saturation, lightness = _lab_to_hsl(L_mean, A_mean, B_mean)
        else:
            hue_mean = base_hue; hue_spread = 25.0
            saturation = max(base_sat, 0.4); lightness = max(base_lit, 0.35)
            chroma_mean = saturation * 55.0; ii = 0.0
    else:
        hue_mean = base_hue; hue_spread = 25.0
        saturation = max(base_sat, 0.4); lightness = max(base_lit, 0.35)
        chroma_mean = saturation * 55.0; ii = 0.0

    parts_data.append({
        "name":              name,
        "color":             color,
        "hue_mean":          round(hue_mean,   2),
        "hue_spread":        round(hue_spread, 2),
        "saturation":        round(saturation, 4),
        "lightness":         round(lightness,  4),
        "chroma_mean":       round(chroma_mean if 'chroma_mean' in dir() else saturation * 55, 2),
        "iridescence_index": ii,
        "ii_level":          _ii_level(ii),
    })

if not parts_data:
    st.info("No parts defined yet.")
    st.stop()

# ── Three.js + GLSL component ─────────────────────────────────────────────────
parts_json = json.dumps(parts_data)
n_cols     = min(len(parts_data), 4)
n_rows     = (len(parts_data) + n_cols - 1) // n_cols
height_px  = n_rows * 420 + 20

VERTEX_SHADER = """
varying vec3 vNormal;
varying vec3 vViewDir;
void main() {
    vNormal  = normalize(normalMatrix * normal);
    vec4 wp  = modelMatrix * vec4(position, 1.0);
    vViewDir = normalize(cameraPosition - wp.xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
uniform float uBaseHue;
uniform float uHueSpread;
uniform float uSaturation;
uniform float uLightness;
uniform vec3  uLightDir;

varying vec3 vNormal;
varying vec3 vViewDir;

vec3 hsl2rgb(float h, float s, float l) {
    h = fract(h);
    float c  = (1.0 - abs(2.0*l - 1.0)) * s;
    float hp = h * 6.0;
    float x  = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
    float m  = l - c * 0.5;
    vec3 rgb;
    if      (hp < 1.0) rgb = vec3(c, x, 0.0);
    else if (hp < 2.0) rgb = vec3(x, c, 0.0);
    else if (hp < 3.0) rgb = vec3(0.0, c, x);
    else if (hp < 4.0) rgb = vec3(0.0, x, c);
    else if (hp < 5.0) rgb = vec3(x, 0.0, c);
    else               rgb = vec3(c, 0.0, x);
    return clamp(rgb + m, 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);

    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    // Thin-film iridescence: hue rotates with viewing angle
    float hue = mod(uBaseHue + (1.0 - NdotV) * uHueSpread, 360.0) / 360.0;
    // Saturation peaks at grazing angles (physically correct for thin-film)
    float sat = clamp(uSaturation * (0.55 + 0.45 * (1.0 - NdotV)), 0.0, 1.0);
    float lit = clamp(uLightness, 0.15, 0.80);

    vec3 iridColor = hsl2rgb(hue, sat, lit);

    float fresnel = pow(1.0 - NdotV, 4.0);
    float spec    = pow(NdotH, 110.0);

    vec3 color = iridColor * 0.18                         // ambient
               + iridColor * NdotL * 0.70                 // diffuse
               + vec3(1.0) * spec * 0.60                  // sharp specular
               + mix(iridColor, vec3(1.0), 0.45)          // fresnel rim
                 * fresnel * 0.55;

    gl_FragColor = vec4(color, 1.0);
}
"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #0e1117;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #fafafa;
  }}
  .grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    padding: 16px;
    justify-content: center;
  }}
  .card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #16181f;
    border: 1px solid #2a2d38;
    border-radius: 14px;
    padding: 14px 14px 16px;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: #4a4d5a; }}
  canvas {{
    border-radius: 50%;
    cursor: grab;
    display: block;
  }}
  canvas:active {{ cursor: grabbing; }}
  .part-name {{
    font-size: 15px;
    font-weight: 700;
    margin-top: 12px;
    letter-spacing: .4px;
    text-transform: capitalize;
  }}
  .metrics {{
    font-size: 11px;
    color: #777;
    margin-top: 6px;
    text-align: center;
    line-height: 1.8;
  }}
  .metrics span {{ color: #aaa; font-weight: 500; }}
  .badge {{
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .3px;
  }}
</style>
</head>
<body>
<div class="grid" id="grid"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
const PARTS = {parts_json};

const VERT = `{VERTEX_SHADER}`;
const FRAG = `{FRAGMENT_SHADER}`;

function iiColor(ii) {{
  if (ii > 8) return '#ff4f4f';
  if (ii > 4) return '#ff9500';
  if (ii > 1) return '#e5c100';
  return '#4caf50';
}}

PARTS.forEach(function(part, idx) {{
  const SIZE = 240;

  // ── Card ──────────────────────────────────────────────────────────────────
  const card = document.createElement('div');
  card.className = 'card';

  const canvas = document.createElement('canvas');
  canvas.width  = SIZE;
  canvas.height = SIZE;
  card.appendChild(canvas);

  const nameEl = document.createElement('div');
  nameEl.className   = 'part-name';
  nameEl.textContent = part.name;
  nameEl.style.color = part.color;
  card.appendChild(nameEl);

  const metricsEl = document.createElement('div');
  metricsEl.className = 'metrics';
  metricsEl.innerHTML =
    'Base hue <span>' + part.hue_mean.toFixed(1) + '°</span>' +
    '&nbsp;&nbsp;Spread <span>±' + part.hue_spread.toFixed(1) + '°</span><br>' +
    'Chroma <span>' + part.chroma_mean.toFixed(1) + '</span>' +
    '&nbsp;&nbsp;L* <span>' + (part.lightness * 100).toFixed(1) + '</span>';
  card.appendChild(metricsEl);

  const badge = document.createElement('div');
  badge.className = 'badge';
  const c = iiColor(part.iridescence_index);
  badge.style.cssText =
    'background:' + c + '22;color:' + c + ';border:1px solid ' + c + '55;';
  badge.textContent = part.ii_level + '  ·  index ' + part.iridescence_index.toFixed(3);
  card.appendChild(badge);

  document.getElementById('grid').appendChild(card);

  // ── Three.js ───────────────────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
  renderer.setSize(SIZE, SIZE);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0e1117, 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0, 3.4);

  const geo = new THREE.SphereGeometry(1, 128, 128);
  const mat = new THREE.ShaderMaterial({{
    uniforms: {{
      uBaseHue:   {{ value: part.hue_mean   }},
      uHueSpread: {{ value: part.hue_spread  }},
      uSaturation:{{ value: part.saturation  }},
      uLightness: {{ value: part.lightness   }},
      uLightDir:  {{ value: new THREE.Vector3(1.5, 1.0, 1.0).normalize() }},
    }},
    vertexShader:   VERT,
    fragmentShader: FRAG,
  }});

  const sphere = new THREE.Mesh(geo, mat);
  scene.add(sphere);

  // ── Mouse / touch drag ─────────────────────────────────────────────────────
  let drag = false, px = 0, py = 0;

  canvas.addEventListener('mousedown',  function(e) {{ drag=true; px=e.clientX; py=e.clientY; }});
  window.addEventListener('mouseup',    function()  {{ drag=false; }});
  canvas.addEventListener('mousemove',  function(e) {{
    if (!drag) return;
    sphere.rotation.y += (e.clientX - px) * 0.009;
    sphere.rotation.x += (e.clientY - py) * 0.009;
    px = e.clientX; py = e.clientY;
  }});
  canvas.addEventListener('touchstart', function(e) {{ drag=true; px=e.touches[0].clientX; py=e.touches[0].clientY; }}, {{passive:true}});
  canvas.addEventListener('touchend',   function()  {{ drag=false; }});
  canvas.addEventListener('touchmove',  function(e) {{
    if (!drag) return;
    sphere.rotation.y += (e.touches[0].clientX - px) * 0.009;
    sphere.rotation.x += (e.touches[0].clientY - py) * 0.009;
    px = e.touches[0].clientX; py = e.touches[0].clientY;
  }}, {{passive:true}});

  // ── Animation loop ─────────────────────────────────────────────────────────
  var t = idx * (Math.PI * 2 / PARTS.length);   // stagger start phase per sphere
  (function loop() {{
    requestAnimationFrame(loop);
    t += 0.013;
    // Light orbits in a tilted ellipse — makes hue shift clearly visible
    mat.uniforms.uLightDir.value.set(
      Math.sin(t) * 2.0,
      Math.cos(t * 0.5) * 0.7 + 0.8,
      Math.cos(t) * 2.0
    ).normalize();
    if (!drag) sphere.rotation.y += 0.003;
    renderer.render(scene, camera);
  }})();
}});
</script>
</body>
</html>
"""

components.html(html, height=height_px, scrolling=False)

# ── Legend ─────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("How to read these spheres"):
    st.markdown("""
**What you are looking at**

Each sphere is rendered using the color measurements taken directly from your video.
The shader simulates **thin-film optical interference**, which is the same physical
process that gives hummingbird feathers their iridescent shimmer.

**Why the hue shifts across the surface**

The shader measures the angle between the surface normal and your viewing direction at every pixel.
When the surface faces you directly (the centre of the sphere), you see the base hue.
As the surface tilts away toward the edges, the hue rotates by the measured spread amount.
This is exactly why iridescent feathers appear to change color as you move around them.

**The orbiting light**

The directional light moves in a tilted ellipse so you can see how the color interacts with
the illumination angle. At grazing angles the Fresnel effect also increases saturation,
which matches the physics of thin-film reflectors.

**How the measurements map to the render**

| Measured value | Shader input | What it controls |
|---|---|---|
| Mean hue | Base hue | The color you see straight-on |
| Hue spread | Hue range | How much the color rotates edge to center |
| Mean chroma | Saturation | How vivid the color appears |
| Mean brightness | Lightness | Overall darkness or lightness of the material |

> These spheres show what a perfect thin-film material with your measured parameters would look like.
> Real feathers also have micro-texture and scattering that this model does not include.
""")
