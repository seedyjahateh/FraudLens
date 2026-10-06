import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fmt } from "../lib/format";
import { fixtureBundle, mockApi, renderApp } from "../test/server";

afterEach(() => vi.unstubAllGlobals());

describe("Overview", () => {
  it("shows the served model's test results with intervals", async () => {
    mockApi();
    renderApp("/");
    expect(await screen.findByRole("heading", { name: /held-out test slice/i })).toBeInTheDocument();
    const prAuc = fixtureBundle.metrics.find(
      (r) => r.model === fixtureBundle.meta.served_label && r.metric === "pr_auc" && r.split === "test",
    );
    expect(screen.getAllByText(fmt.num(prAuc?.value ?? NaN)).length).toBeGreaterThan(0);
    expect(screen.getByText(/Test PR-AUC with 95% intervals/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /confusion matrix/i })).toBeInTheDocument();
  });

  it("shows the finding only when a model beat the served one on test", async () => {
    mockApi();
    renderApp("/");
    await screen.findByRole("heading", { name: /held-out test slice/i });
    const finding = screen.queryByText(/not the best on test/i);
    expect(Boolean(finding)).toBe(fixtureBundle.finding.better_unselected.length > 0);
  });

  it("offers a table view for every chart", async () => {
    mockApi();
    renderApp("/");
    await screen.findByRole("heading", { name: /held-out test slice/i });
    const tableButtons = screen.getAllByRole("radio", { name: /table/i });
    expect(tableButtons.length).toBeGreaterThanOrEqual(3);
    await userEvent.click(tableButtons[0]!);
    expect(await screen.findByText("Interpolated precision by recall")).toBeInTheDocument();
  });

  it("renders an honest error when the bundle is missing", async () => {
    mockApi({
      "/api/dashboard": () => ({ status: 404, body: { detail: "no dashboard bundle for model version x" } }),
    });
    renderApp("/");
    expect(await screen.findByText(/Nothing to show yet/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("renders a contract error instead of crashing on bad data", async () => {
    mockApi({ "/api/dashboard": () => ({ body: { schema_version: 1 } }) });
    renderApp("/");
    expect(await screen.findByText(/unexpected shape/i)).toBeInTheDocument();
  });
});

describe("Threshold explorer", () => {
  it("starts at the served threshold on validation", async () => {
    mockApi();
    renderApp("/threshold");
    await screen.findByRole("heading", { name: /threshold & cost explorer/i });
    expect(screen.getByRole("radio", { name: /validation/i })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("slider", { name: /decision threshold/i })).toHaveAttribute(
      "aria-valuenow",
      String(fixtureBundle.sweeps.validation.chosen_index),
    );
    expect(screen.queryByText(/test view is illustrative/i)).not.toBeInTheDocument();
  });

  it("warns that the test view is illustrative", async () => {
    mockApi();
    renderApp("/threshold");
    await userEvent.click(await screen.findByRole("radio", { name: /^test/i }));
    expect(screen.getByText(/test view is illustrative/i)).toBeInTheDocument();
  });

  it("moves with the keyboard and resets to the served threshold", async () => {
    mockApi();
    renderApp("/threshold");
    const slider = await screen.findByRole("slider", { name: /decision threshold/i });
    slider.focus();
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    const moved = Number(slider.getAttribute("aria-valuenow"));
    expect(moved).toBe(fixtureBundle.sweeps.validation.chosen_index + 2);
    await userEvent.click(screen.getByRole("button", { name: /served threshold/i }));
    expect(slider).toHaveAttribute("aria-valuenow", String(fixtureBundle.sweeps.validation.chosen_index));
  });

  it("recomputes cost when the review cost changes", async () => {
    mockApi();
    renderApp("/threshold");
    const input = await screen.findByLabelText(/false alarm/i);
    await userEvent.clear(input);
    await userEvent.type(input, "250");
    expect(screen.getByRole("button", { name: /reset to config\.yaml/i })).toBeInTheDocument();
  });
});

describe("Playground", () => {
  it("scores a preset and shows the decision and contributions", async () => {
    const fetchMock = mockApi();
    renderApp("/playground");
    await userEvent.click(await screen.findByRole("button", { name: /suspicious/i }));
    expect(await screen.findByText(/flag for review/i)).toBeInTheDocument();
    const list = screen.getByRole("list", { name: /top contributing features/i });
    expect(within(list).getByText("V14")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/score", expect.objectContaining({ method: "POST" }));
    expect(screen.getByText("Suspicious example", { selector: "span" })).toBeInTheDocument();
  });

  it("blocks invalid input before calling the API", async () => {
    const fetchMock = mockApi();
    renderApp("/playground");
    const amount = await screen.findByLabelText("Amount");
    await userEvent.clear(amount);
    await userEvent.type(amount, "-5");
    await userEvent.click(screen.getByRole("button", { name: /score transaction/i }));
    expect(await screen.findByText("Must be ≥ 0")).toBeInTheDocument();
    expect(amount).toHaveAttribute("aria-invalid", "true");
    expect(fetchMock).not.toHaveBeenCalledWith("/score", expect.anything());
  });

  it("highlights fields the server rejects", async () => {
    mockApi({
      "/score": () => ({
        status: 422,
        body: {
          detail: "invalid transaction",
          errors: [{ field: "V7", message: "Input should be a finite number", type: "finite_number" }],
        },
      }),
    });
    renderApp("/playground");
    await userEvent.click(await screen.findByRole("button", { name: /score transaction/i }));
    expect(await screen.findByText("Input should be a finite number")).toBeInTheDocument();
    expect(screen.getByLabelText("V7")).toHaveAttribute("aria-invalid", "true");
  });

  it("deep-links to a preset and scores it", async () => {
    mockApi();
    renderApp("/playground?preset=typical");
    expect(await screen.findByText(/flag for review/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Typical transaction" })).toBeInTheDocument();
  });
});

describe("Model & service", () => {
  it("shows provenance, live stats and the search", async () => {
    mockApi();
    renderApp("/model");
    expect(await screen.findByRole("heading", { name: /model & service/i, level: 1 })).toBeInTheDocument();
    expect(await screen.findByText(fixtureBundle.meta.version, { selector: "span" })).toBeInTheDocument();
    expect(await screen.findByText("9.8 ms")).toBeInTheDocument();
    expect(screen.getByText(/how the model was chosen/i)).toBeInTheDocument();
    expect(screen.getByText(/No load test recorded/i)).toBeInTheDocument();
  });
});

describe("Shell", () => {
  it("reports a live service and toggles the theme", async () => {
    mockApi();
    renderApp("/");
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/live/i));
    const toggle = screen.getByRole("button", { name: /switch to light theme/i });
    await userEvent.click(toggle);
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("opens the command menu with Ctrl+K", async () => {
    mockApi();
    renderApp("/");
    await screen.findByRole("heading", { name: /held-out test slice/i });
    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    expect(await screen.findByPlaceholderText(/jump to a view/i)).toBeInTheDocument();
  });

  it("redirects unknown routes to the overview", async () => {
    mockApi();
    renderApp("/nope");
    expect(await screen.findByRole("heading", { name: /held-out test slice/i })).toBeInTheDocument();
  });
});
