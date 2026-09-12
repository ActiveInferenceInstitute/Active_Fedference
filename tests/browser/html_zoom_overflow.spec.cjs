const path = require("node:path");
const { pathToFileURL } = require("node:url");

const { expect, test } = require("@playwright/test");

test.setTimeout(90000);

async function settleReader(page) {
  if (await page.locator("script[data-template-mathjax-config]").count()) {
    await page.waitForFunction(() => typeof window.MathJax?.startup?.document?.rerenderPromise === "function");
    await page.evaluate(async () => {
      await window.MathJax.startup.promise;
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      let reflow;
      do {
        reflow = window.templateMathJaxReflowPromise;
        await reflow;
      } while (reflow !== window.templateMathJaxReflowPromise);
      if (window.templateMathJaxReflowError) throw new Error(window.templateMathJaxReflowError);
    });
  }
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all([...document.images].map((image) => image.decode()));
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
}

const ROOT = process.cwd();
const defaultPage = path.join(
  ROOT,
  "output",
  "web",
  "manuscript__30_supplement_notation.html",
);
const HTML_PAGES = process.env.FEDFERENCE_HTML_PAGES
  ? JSON.parse(process.env.FEDFERENCE_HTML_PAGES)
  : [defaultPage];

if (!Array.isArray(HTML_PAGES) || HTML_PAGES.length === 0) {
  throw new Error("FEDFERENCE_HTML_PAGES must be a non-empty JSON array");
}
if (HTML_PAGES.some((page) => typeof page !== "string" || page.length === 0)) {
  throw new Error("FEDFERENCE_HTML_PAGES entries must be non-empty strings");
}
if (new Set(HTML_PAGES).size !== HTML_PAGES.length) {
  throw new Error("FEDFERENCE_HTML_PAGES must not contain duplicate paths");
}

for (const htmlPage of HTML_PAGES) {
  const pageName = path.basename(htmlPage);
  test(`${pageName} keeps desktop figure edges visible`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(pathToFileURL(htmlPage).href, { waitUntil: "domcontentloaded" });
    await settleReader(page);
    const clipped = await page.locator("figure img").evaluateAll((images) => {
      const findings = [];
      for (const image of images) {
        const bounds = image.getBoundingClientRect();
        if (bounds.left < -1 || bounds.right > innerWidth + 1) {
          findings.push({ image: image.getAttribute("src"), ancestor: "viewport" });
        }
        for (let ancestor = image.parentElement; ancestor; ancestor = ancestor.parentElement) {
          const overflow = getComputedStyle(ancestor).overflowX;
          if (!["hidden", "clip", "auto", "scroll"].includes(overflow)) continue;
          const clip = ancestor.getBoundingClientRect();
          if (bounds.left < clip.left - 1 || bounds.right > clip.right + 1) {
            findings.push({ image: image.getAttribute("src"), ancestor: ancestor.tagName, overflow });
          }
        }
      }
      return findings;
    });
    expect(clipped).toEqual([]);
  });
  test(`${pageName} reflows completed equations after live zoom changes`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(pathToFileURL(htmlPage).href, { waitUntil: "domcontentloaded" });
    await settleReader(page);
    const source = await page.evaluate(() => window.MathJax?.startup?.document
      ?.getMathItemsWithin(document.body).map((item) => item.math) || []);
    for (const width of [640, 320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await settleReader(page);
      const geometry = await page.evaluate(() => ({
        viewportWidth: document.documentElement.clientWidth,
        bodyScrollWidth: document.body.scrollWidth,
        source: window.MathJax?.startup?.document
          ?.getMathItemsWithin(document.body).map((item) => item.math) || [],
        clippedMath: [...document.querySelectorAll("mjx-math")]
          .filter((element) => !element.closest(".table-scroll"))
          .filter((element) => {
            for (let ancestor = element; ancestor; ancestor = ancestor.parentElement) {
              const style = getComputedStyle(ancestor);
              if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
            }
            const rect = element.getBoundingClientRect();
            return rect.width > 0 && (rect.left < -1 || rect.right > innerWidth + 1);
          })
          .map((element) => ({ text: element.textContent, bounds: element.getBoundingClientRect().toJSON() })),
      }));
      expect(geometry.source).toEqual(source);
      // Desktop figures intentionally extend beyond the centered text column.
      // The reader viewport, not that narrower column, is the clipping boundary.
      expect(geometry.bodyScrollWidth).toBeLessThanOrEqual(geometry.viewportWidth + 1);
      expect(geometry.clippedMath).toEqual([]);
    }
  });
  for (const zoom of [2, 4]) {
    test(`${pageName} keeps overflow local at ${zoom * 100}% zoom`, async ({ browser }) => {
      // Browser zoom reduces the CSS viewport while retaining the same physical
      // pixel budget. This context is the deterministic Playwright equivalent:
      // 1280 physical pixels become 640 CSS pixels at 200% and 320 at 400%.
      const context = await browser.newContext({
        viewport: { width: Math.floor(1280 / zoom), height: Math.floor(900 / zoom) },
        deviceScaleFactor: zoom,
      });
      const page = await context.newPage();
      await page.goto(pathToFileURL(htmlPage).href, { waitUntil: "domcontentloaded" });
      await settleReader(page);

      const geometry = await page.evaluate(() => {
        const root = document.documentElement;
        const body = document.body;
        const viewportWidth = window.innerWidth;
        const tables = [...document.querySelectorAll(".table-scroll")].map((wrapper) => {
          const style = getComputedStyle(wrapper);
          return {
            clientWidth: wrapper.clientWidth,
            scrollWidth: wrapper.scrollWidth,
            overflowX: style.overflowX,
            right: wrapper.getBoundingClientRect().right,
          };
        });
        const offViewportElements = [...document.querySelectorAll("body *")]
          .filter((element) => !element.closest(".table-scroll"))
          .filter((element) => {
            const style = getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden") {
              return false;
            }
            const rect = element.getBoundingClientRect();
            return rect.width > 0 && (rect.left < -1 || rect.right > viewportWidth + 1);
          })
          .map((element) => ({
            tag: element.tagName.toLowerCase(),
            id: element.id,
            className: String(element.className || ""),
            left: element.getBoundingClientRect().left,
            right: element.getBoundingClientRect().right,
          }));
        const nonTableHorizontalScrollers = [...document.querySelectorAll("body *")]
          .filter((element) => !element.closest(".table-scroll"))
          .filter((element) => {
            const style = getComputedStyle(element);
            return (
              ["auto", "scroll"].includes(style.overflowX) &&
              element.scrollWidth > element.clientWidth + 1
            );
          })
          .map((element) => ({
            tag: element.tagName.toLowerCase(),
            id: element.id,
            className: String(element.className || ""),
            clientWidth: element.clientWidth,
            scrollWidth: element.scrollWidth,
          }));
        return {
          bodyClientWidth: body.clientWidth,
          bodyScrollWidth: body.scrollWidth,
          rootClientWidth: root.clientWidth,
          rootScrollWidth: root.scrollWidth,
          viewportWidth,
          tableElementCount: document.querySelectorAll("table").length,
          tables,
          offViewportElements,
          nonTableHorizontalScrollers,
        };
      });

      expect(geometry.rootScrollWidth).toBeLessThanOrEqual(geometry.rootClientWidth + 1);
      expect(geometry.bodyScrollWidth).toBeLessThanOrEqual(geometry.bodyClientWidth + 1);
      expect(geometry.offViewportElements).toEqual([]);
      expect(geometry.nonTableHorizontalScrollers).toEqual([]);
      expect(geometry.tables.length).toBe(geometry.tableElementCount);
      for (const table of geometry.tables) {
        expect(["auto", "scroll"]).toContain(table.overflowX);
        expect(table.right).toBeLessThanOrEqual(geometry.viewportWidth + 1);
      }

      if (zoom === 2) {
        // Retain real keyboard/focus evidence in the same browser run rather
        // than treating static markup as proof of operability.
        await page.keyboard.press("Tab");
        const firstFocused = await page.evaluate(() => ({
          className: document.activeElement?.className || "",
          href: document.activeElement?.getAttribute("href") || "",
        }));
        expect(String(firstFocused.className).split(/\s+/)).toContain("skip-link");
        expect(firstFocused.href).toBe("#main-content");
        await page.keyboard.press("Enter");
        await expect(page.locator("#main-content")).toBeFocused();

        const fullSizeLinks = page.locator("a.figure-full-size-link");
        for (let index = 0; index < await fullSizeLinks.count(); index += 1) {
          const link = fullSizeLinks.nth(index);
          const accessibleName = await link.getAttribute("aria-label");
          const contextualName = /^Open full[- ]size (?:Figure\s+\d+|figure),\s*\S.+/i;
          expect(accessibleName).toMatch(contextualName);
          await expect(link).toHaveAccessibleName(contextualName);
          expect(await link.evaluate((element) => element.tabIndex)).toBe(0);
          await link.focus();
          await expect(link).toBeFocused();
        }

        const details = page.locator("details.figure-long-description");
        for (let index = 0; index < await details.count(); index += 1) {
          const disclosure = details.nth(index);
          const detailsId = await disclosure.getAttribute("id");
          expect(detailsId).toBeTruthy();
          await expect(disclosure).not.toHaveAttribute("open", "");
          const summary = disclosure.locator(":scope > summary");
          await expect(summary).toHaveCount(1);
          await summary.focus();
          await expect(summary).toBeFocused();
          await page.keyboard.press("Enter");
          await expect(disclosure).toHaveAttribute("open", "");
          const associatedImages = page.locator(`img[aria-details="${detailsId}"]`);
          await expect(associatedImages).toHaveCount(1);
          await page.keyboard.press("Space");
          await expect(disclosure).not.toHaveAttribute("open", "");
        }
      }

      await context.close();
    });
  }
}
