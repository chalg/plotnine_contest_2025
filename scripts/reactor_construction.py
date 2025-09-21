import os
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_segment, theme_minimal, theme, element_text, element_rect,
    element_blank, scale_x_datetime, scale_color_manual, labs, guides, guide_legend,
    annotate, geom_vline, geom_text, geom_label
)

# ------------- Helpers ----------------

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rough equivalent of janitor::clean_names() in R.
    Lowercase, remove punctuation, replace spaces with underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[^\w\s]", "", regex=True)
          .str.replace(r"\s+", "_", regex=True)
          .str.lower()
    )
    return df

def to_datetime_cols(df: pd.DataFrame, contains: str = "date") -> pd.DataFrame:
    df = df.copy()
    for c in [c for c in df.columns if contains in c]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def to_numeric_cols(df: pd.DataFrame, contains: str = "power") -> pd.DataFrame:
    df = df.copy()
    for c in [c for c in df.columns if contains in c]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def reorder_categorical_by_series(desc_keys: pd.Series, labels: pd.Series) -> pd.Categorical:
    """
    Mimics fct_reorder(unit_name, -con_seq) by explicitly setting the category order.
    desc_keys: numeric keys (e.g., con_seq), higher first
    labels:    corresponding labels (e.g., unit_name strings)
    """
    order = labels.iloc[np.argsort(-desc_keys.values)].tolist()
    return pd.Categorical(labels, categories=order, ordered=True)

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ------------- Read data ----------------

# source: https://pris.iaea.org/pris/   [Reactor Specification_mar_25.xlsx output from there - requires an account]
reactors = pd.read_excel("data/Reactor Specification_mar_25.xlsx")
reactors = clean_names(reactors)

# convert "date" columns to datetime, "power" columns to numeric
reactors = to_datetime_cols(reactors, "date")
reactors = to_numeric_cols(reactors, "power")

# Derived fields
reactors["build_days"] = (reactors["grid_date"] - reactors["construction_date"]).dt.days
reactors["days_mw"] = reactors["build_days"] / reactors["original_design_net_electrical_power"]
reactors["age"] = np.round((pd.Timestamp.today() - reactors["grid_date"]).dt.days / 365.0, 2)

# --- Quick review ---
reactors.info()

(
    reactors
    .loc[(reactors["days_mw"].notna()) & (reactors["status"] == "Operational")]
    .groupby(["country", "status", "type"], dropna=False)["days_mw"]
    .median()
    .round(3)
    .reset_index(name="med_days_mw")
    .drop(columns=["status"])
    .sort_values("med_days_mw")
    .head(10)
    # .pipe(print)  # uncomment if you want to print
)

# Subset + recode
sub1 = (
    reactors
    .loc[
        ~reactors["status"].isin(["Under Construction", "Planned"]) &
        reactors["days_mw"].notna()
    ]
    .copy()
)

# Map to categories: more readable country names
# Most of the dominant reactor construction countries
country_map1 = {
    "UNITED STATES OF AMERICA": "United States",
    "FRANCE"                  :"France",
    "RUSSIA"                  :"Russia",
    "JAPAN"                   :"Japan",
    "KOREA, REPUBLIC OF"      :"South Korea",
    "CANADA"                  :"Canada",
    "CHINA"                   :"China"
}
sub1["country_cat"] = sub1["country"].map(country_map1).fillna("Other")

# Order by grid date then unit name (like arrange(grid_date, unit_name))
sub1 = sub1.sort_values(["grid_date", "unit_name"], ascending=[True, True]).copy()
sub1["con_seq"] = np.arange(1, len(sub1) + 1)

# Reorder factor so that larger con_seq (latest) is at the top
sub1["unit_name"] = reorder_categorical_by_series(sub1["con_seq"], sub1["unit_name"])

# Colors for this plot
colors1 = {
    "China"         : "#FF0000",
    "Canada"        : "#9A32CD",
    "France"        : "#00A08A",
    "Japan"         : "#F2AD00",
    "Russia"        : "#F98400",
    "South Korea"   : "#5BBCD6",
    "United States" : "#9986A5", 
    "Other"         : "#7F7F7F" 
}

# Inset table data
inset_tbl1 = (
    sub1
    .groupby("country_cat", dropna=False, as_index=False)
    .agg(**{
        "Median Days per MW": ("days_mw", lambda s: round(float(np.median(s.dropna())), 3)),
        "Median Construction Years": ("build_days", lambda s: round(float(np.median(s.dropna()))/365.0, 3))
    })
    .rename(columns={"country_cat": "Country"})
    .sort_values("Median Days per MW")
    
)

# Add ALL row (across entire subset)
inset_tbl1 = pd.concat([
    inset_tbl1,
    pd.DataFrame([{
        "Country": "ALL",
        "Median Days per MW": round(float(np.median(sub1["days_mw"].dropna())), 3),
        "Median Construction Years": round(float(np.median(sub1["build_days"].dropna()))/365.0, 3)
    }])
], ignore_index=True)

# Helper to put the disaster labels roughly near the top of the list
cats1 = list(sub1["unit_name"].cat.categories)
y_for_event_text = cats1[len(cats1)-20] if len(cats1) else None
y_for_table_text = cats1[len(cats1)-50] if len(cats1) else None  # put the table a bit lower

# Compose a simple text table (since ggpmisc::annotate(table) has no direct analogue in plotnine)
def as_block_text(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = f"{cols[0]:<12} {cols[1]:>18} {cols[2]:>25}"
    lines = [header]
    for _, r in df.iterrows():
        lines.append(f"{str(r[cols[0]])[:13]:<13} {r[cols[1]]:>18.3f} {r[cols[2]]:>25.3f}")
    return "\n".join(lines)

inset_block1 = as_block_text(inset_tbl1)


# --- Build Plot 1 (fixed) ---
p1 = (
    ggplot(sub1)
    + geom_segment(
        aes(x='construction_date', xend='grid_date', y='unit_name', yend='unit_name', color='country_cat'),
        size=1.5
    )
    # Disaster verticals
    + geom_vline(xintercept=pd.Timestamp("1979-03-28"), linetype="dashed", color="cadetblue")
    + geom_vline(xintercept=pd.Timestamp("1986-04-26"), linetype="dashed", color="cadetblue")
    + geom_vline(xintercept=pd.Timestamp("2011-03-11"), linetype="dashed", color="cadetblue")
)

# Event text (annotate has inherit_aes=False by default)
if y_for_event_text is not None:
    p1 = (
        p1
        + annotate("text",
                   x=pd.Timestamp("1979-03-28") - pd.Timedelta(days=900),
                   y=y_for_event_text, label="Three Mile Island",
                   size=9, color="#404040", ha="left")
        + annotate("text",
                   x=pd.Timestamp("1986-04-26") + pd.Timedelta(days=100),
                   y=y_for_event_text, label="Chernobyl",
                   size=9, color="#404040", ha="left")
        + annotate("text",
                   x=pd.Timestamp("2011-03-11") + pd.Timedelta(days=100),
                   y=y_for_event_text, label="Fukushima",
                   size=9, color="#404040", ha="left")
    )

# Inset "table" as a text block (left-aligned, larger, monospace)
if y_for_table_text is not None:
    p1 = p1 + geom_label(
        aes(x="x", y="y", label="label"),
        data=pd.DataFrame({
            "x": [pd.Timestamp("1987-01-01")],   # adjust if you want it farther right/left
            "y": [y_for_table_text],
            "label": [inset_block1]
        }),
        inherit_aes=False,
        # Appearance & legibility
        size=10,                 # was 7 -> bigger text
        lineheight=1.15,         # a touch more spacing between lines
        family="DejaVu Sans Mono",  # monospace helps columns align (fallback: "Courier New")
        color="#333333",
        fill="white",
        alpha=0.95,
        label_size=0.25,         # thin border
        # Alignment: top-left corner at (x, y)
        ha="left",
        va="top"
        # Optional (plotnine ≥0.10): label_padding=0.35, label_r=0.05
    )

# Label Watts Bar reactors (must NOT inherit aes)
labels1 = sub1[sub1["unit_name"].isin(["WATTS BAR-1", "WATTS BAR-2"])].copy()
if not labels1.empty:
    labels1["x_label"] = labels1["grid_date"] + pd.to_timedelta(1150, unit="D")
    p1 = p1 + geom_text(
        aes(x="x_label", y="unit_name", label="unit_name"), 
        data=labels1, color="#404040", size=7, nudge_x=1,
        inherit_aes=False
    )

# Theming & scales
p1 = (
    p1
    + theme_minimal()
    + scale_color_manual(values=colors1)
    + scale_x_datetime(
        limits=(pd.Timestamp("1950-01-01"), pd.Timestamp("2025-06-30")),
        date_breaks="10 years", date_labels="%Y"
    )
    + labs(
        title="Nuclear Reactor Construction (first concrete to grid connection)",
        subtitle=(
            "Reactors ordered by grid connection. Note: Watts Bar project (Tennessee, US) was put on hold for 22 years."
            ),
        x="Time", y="Reactor", color="Country",
        caption="Dataviz: @GrantChalmers | Source: https://pris.iaea.org"
    )
    + guides(color=guide_legend(ncol=1))
    + theme(
        plot_title=element_text(size=11, weight="bold"),
        plot_subtitle=element_text(size=9.5, style="italic", color="#404040"),
        axis_title_y=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks_major_y=element_blank(),
        axis_text_x=element_text(size=9, angle=0),
        axis_title=element_text(size=11),
        legend_position=(0.555, 0.73),
        legend_background=element_rect(fill='white', color='lightgrey'),
        plot_caption=element_text(size=8.5, color="#7f7f7f", style="italic"),
        legend_title=element_text(size=10.5, family="DejaVu Sans", color="#333333"),
        legend_text=element_text(size=10, family="DejaVu Sans", color="#333333"),
        plot_background=element_rect(fill='antiquewhite', color='antiquewhite'),
        panel_background=element_rect(fill='snow')
    )
)

# Save
out1 = "images/reactor_cons_by_grid_con_mar25_long_labels.png"
ensure_dir(out1)
p1.save(out1, width=10, height=13, dpi=300)
print(f"Saved: {out1}")

