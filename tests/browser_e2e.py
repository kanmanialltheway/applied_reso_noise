"""Optional browser test: drives the real UI in headless Chromium (needs `pip install playwright` + `playwright install chromium`).

    python tests/browser_e2e.py            # starts its own server on a temporary database
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from resonoise import audio_io, synthetic  # noqa: E402

PORT = int(os.environ.get("E2E_PORT", "5077"))
URL = f"http://127.0.0.1:{PORT}"
SHOTS = os.environ.get("E2E_SHOTS")


def make_wavs(tmp):
    sc = synthetic.noise_scene("speech", np.random.default_rng(1), 8.0)
    t = np.arange(len(sc["mix"])) / 16000
    hum = 0.25 * (np.sin(2 * np.pi * 120 * t) + 0.6 * np.sin(2 * np.pi * 240 * t) + 0.4 * np.sin(2 * np.pi * 360 * t))
    mix = sc["mix"] * 0.5 + hum + 0.01 * np.random.default_rng(1).standard_normal(t.size)
    mix = mix / np.max(np.abs(mix)) * 0.9
    p = os.path.join(tmp, "kitchen_hum.wav")
    open(p, "wb").write(audio_io.wav_bytes(mix, 16000))
    bad = os.path.join(tmp, "notes.txt")
    open(bad, "w").write("this is not audio")
    return p, bad


def wait_server():
    for _ in range(60):
        try:
            urllib.request.urlopen(URL + "/api/health", timeout=1)
            return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("server did not start")


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


def canvas_has_ink(page, selector):
    return page.evaluate("""(sel) => { const c = document.querySelector(sel); if (!c) return false;
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; const seen = new Set();
        for (let i = 0; i < d.length; i += 4 * 17) seen.add(d[i] + ',' + d[i+1] + ',' + d[i+2] + ',' + d[i+3]); return seen.size > 8; }""", selector)


def main():
    from playwright.sync_api import sync_playwright
    tmp = tempfile.mkdtemp()
    wav, bad = make_wavs(tmp)
    env = dict(os.environ, RESONOISE_DATA=os.path.join(tmp, "data"))
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "run.py"), "--port", str(PORT)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream",
                                              "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={wav}"])
            ctx = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True, permissions=["microphone"])
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "fonts.g" not in m.text and "Failed to load resource" not in m.text else None)
            shot = (lambda n: page.screenshot(path=os.path.join(SHOTS, n + ".png"), full_page=True)) if SHOTS else (lambda n: None)

            page.set_default_timeout(15000)
            page.goto(URL)
            page.wait_for_selector(".stepper")
            check(page.locator(".step-btn[aria-current=step]").inner_text().startswith("1"), "studio opens on the input step")
            check(page.locator('[data-step="3"]').is_disabled(), "later steps are locked before any audio")
            shot("1_input")

            # invalid file
            page.set_input_files("#file", bad)
            page.wait_for_selector(".toast.error")
            check("not a recognised audio format" in page.locator(".toast.error").first.inner_text(), "unsupported file shows a clear error")

            # upload real file -> analysis
            page.set_input_files("#file", wav)
            page.wait_for_selector(".step-btn[aria-current=step]:has-text('Spectrum')", timeout=30000)
            page.wait_for_selector("canvas")
            time.sleep(0.8)
            check(page.locator("canvas").count() >= 3, "waveform, spectrum and spectrogram canvases are drawn")
            check(all(canvas_has_ink(page, f"#panel .card:nth-of-type({i}) canvas") for i in (1, 2, 3)), "charts contain real drawn data")
            check("120" in page.locator("#panel").inner_text(), "the 120 Hz hum is reported in the analysis")
            check("Exact" in page.locator(".facts").first.inner_text(), "Parseval check is reported")
            shot("2_spectrum")

            # chart interactions
            page.locator("#panel .card:nth-of-type(2) canvas").scroll_into_view_if_needed()
            box = page.locator("#panel .card:nth-of-type(2) canvas").bounding_box()
            page.mouse.move(box["x"] + 300, box["y"] + 100)
            check(page.locator(".chart-tip:visible").count() >= 1, "spectrum hover shows a readout")
            page.mouse.move(box["x"] + 200, box["y"] + 100)
            page.mouse.down(); page.mouse.move(box["x"] + 330, box["y"] + 100); page.mouse.up()
            check(page.locator("#panel .card:nth-of-type(2) canvas").count() == 1, "drag-zoom on the spectrum works without errors")

            # playback
            page.locator('[data-act="toggle"]').first.click()
            time.sleep(1.2)
            t = page.evaluate("document.querySelector('.time').textContent")
            check(not t.startswith("0:00 / 0:00") and not t.startswith("0:00 / 0:0"), f"audio plays and the clock moves ({t})")
            page.locator('[data-act="toggle"]').first.click()

            # source
            page.click("#next")
            page.wait_for_selector(".verdict")
            check(page.locator(".verdict .label").inner_text().strip() != "", "a noise source is identified")
            check(page.locator(".prob-row").count() >= 5, "class probabilities are listed")
            shot("3_source")
            page.click("#next")
            page.wait_for_selector(".filter-layout")
            time.sleep(0.8)
            check(page.locator("#tones input").count() >= 2, "detected tones are offered as notch candidates")
            shot("4_filter")

            # change a control -> profile becomes custom, preview updates
            page.fill("#add-hz", "1500")
            page.click("#add-btn")
            page.wait_for_selector(".filter-layout")
            check(page.locator("#profile").input_value() == "custom", "editing switches the profile to custom")
            page.click("#apply")
            page.wait_for_selector(".step-btn[aria-current=step]:has-text('Result')", timeout=30000)
            time.sleep(1.0)
            txt = page.locator("#panel").inner_text()
            check("noise floor dropped" in txt.lower() or "overall level changed" in txt.lower(), "result headline summarises the effect")
            check(page.locator(".metric").count() >= 7, "evaluation metrics are shown")
            shot("5_result")

            # A/B + download
            page.click('[data-mode="processed"]')
            check(page.locator('[data-mode="processed"]').get_attribute("aria-pressed") == "true", "A/B switches to the processed audio")
            with page.expect_download() as dl:
                page.click('[data-dl*="processed"][data-dl*="download=1"]')
            path = dl.value.path()
            dec = audio_io.decode_audio(open(path, "rb").read(), "p.wav")
            orig = audio_io.decode_audio(open(wav, "rb").read(), "o.wav")
            check(len(dec.samples) == len(orig.samples) and not np.allclose(dec.samples, orig.samples), "downloaded WAV is valid and differs from the input")
            spec = lambda x: np.abs(np.fft.rfft(x * np.hanning(len(x))))
            f = np.fft.rfftfreq(len(dec.samples), 1 / dec.sample_rate)
            k = np.argmin(np.abs(f - 120))
            drop = 20 * np.log10(spec(dec.samples)[k - 2:k + 3].max() / spec(orig.samples)[k - 2:k + 3].max())
            check(drop < -15, f"the 120 Hz hum is {abs(drop):.0f} dB lower in the downloaded file")

            # library + method
            page.click('a[data-route="library"]')
            page.wait_for_selector(".lib-item")
            page.wait_for_selector("[data-pact=view]")
            check(page.locator(".lib-item").count() == 1, "the library lists the stored recording")
            shot("6_library")
            page.click("[data-pact=view]")
            page.wait_for_selector(".step-btn[aria-current=step]:has-text('Result')")
            check(page.locator(".metric").count() >= 7, "a stored result can be reopened from the library")
            page.click('a[data-route="method"]')
            page.wait_for_selector(".eq-list")
            check(page.locator("math").count() == 5, "the five equations are rendered")
            shot("7_method")

            # sample scene + microphone
            page.click('a[data-route="studio"]')
            page.wait_for_selector(".stepper")
            page.click("#new-rec") if page.locator("#new-rec").count() else None
            page.wait_for_selector("[data-sample]")
            page.click('[data-sample="speech_excavator"]')
            page.wait_for_selector(".step-btn[aria-current=step]:has-text('Spectrum')", timeout=30000)
            check("Sample scene" in page.locator(".current-bar").inner_text(), "sample scenes load and analyse")
            page.click("#new-rec")
            page.wait_for_selector("#rec-start")
            page.click("#rec-start")
            page.wait_for_selector("#rec-stop:visible")
            time.sleep(3.0)
            page.click("#rec-stop")
            page.wait_for_selector(".step-btn[aria-current=step]:has-text('Spectrum')", timeout=30000)
            check("kHz" in page.locator(".current-bar").inner_text() and page.evaluate("document.querySelector('.current-bar .name').textContent") == "recording",
                  "microphone recording is captured, uploaded and analysed")

            # mobile layout
            page.set_viewport_size({"width": 390, "height": 800})
            time.sleep(0.5)
            check(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "no horizontal overflow on a phone-sized screen")
            shot("8_mobile")
            check(not errors, f"no console or page errors {errors[:3]}")
            browser.close()
    except Exception as exc:
        print("ABORTED:", type(exc).__name__, str(exc).splitlines()[0])
        try:
            print("  console/page errors:", errors[:5])
            print("  toasts:", page.locator("#toasts").inner_text())
            print("  hash:", page.evaluate("location.hash"), "| studio hidden:", page.evaluate("document.querySelector('#view-studio').hidden"))
        except Exception:
            pass
        check.failed += 1
    finally:
        srv.terminate()
    print("\n%s" % ("ALL BROWSER CHECKS PASSED" if not check.failed else f"{check.failed} CHECK(S) FAILED"))
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
