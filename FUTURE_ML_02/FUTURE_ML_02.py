"""
FUTURE_ML_02 — Support Ticket Classification & Priority Assignment
A professional ML pipeline for automated customer support ticket triage.

Run: py FUTURE_ML_02.py
"""

import sys, os, re, random, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

random.seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

# ─── Professional Corporate Theme ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",      # Very light gray background
    "axes.facecolor":   "#FFFFFF",      # White axes
    "axes.edgecolor":   "#DEE2E6",      # Soft borders
    "axes.labelcolor":  "#495057",      # Dark gray labels
    "text.color":       "#212529",      # Almost black text
    "xtick.color":      "#6C757D",      # Muted ticks
    "ytick.color":      "#6C757D",
    "grid.color":       "#E9ECEF",      # Very subtle grid
    "grid.linewidth":   0.8,
    "grid.alpha":       1.0,
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "600",          # Semi-bold titles
    "axes.labelsize":   10,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#DEE2E6",
    "legend.labelcolor":"#495057",
    "figure.dpi":       200,            # High resolution
})

# ─── Professional Color Palette (Corporate) ──────────────────────────────────
CORP_COLORS = {
    "primary":   "#0D6EFD",   # Corporate Blue
    "secondary": "#6C757D",   # Slate Gray
    "success":   "#198754",   # Emerald Green
    "warning":   "#FFC107",   # Amber
    "danger":    "#DC3545",   # Crimson
    "info":      "#0DCAF0",   # Teal
    "dark":      "#212529",   # Dark
}
PALETTE_SEQ = [CORP_COLORS["primary"], CORP_COLORS["info"], CORP_COLORS["secondary"], CORP_COLORS["dark"]]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATASET
# Each category has 25 unique templates. We split templates 80/20 to ensure
# the test set contains truly unseen sentence structures (no memorisation).
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "Billing": [
        "I was charged twice for my subscription this month.",
        "My invoice shows an incorrect amount for my plan.",
        "I cancelled my account but was still charged.",
        "Can you send me a detailed invoice for the past quarter?",
        "My credit card was declined during subscription renewal.",
        "There is an unexpected charge of $49.99 on my statement.",
        "I signed up for the free trial but was charged immediately.",
        "My company requires a VAT invoice for the latest payment.",
        "My bank shows a charge from your company but I never enrolled.",
        "The promotional discount promised at signup is missing from my bill.",
        "My subscription price increased without any prior notification.",
        "My account was closed months ago but I keep getting charged.",
        "My promotional code is not being applied at checkout.",
        "I submitted a refund request two weeks ago with no response.",
        "My auto-renewal failed and the account was suspended.",
        "The payment was processed but my account still shows as unpaid.",
        "I downgraded my plan but was billed at the higher rate.",
        "I need to update my billing address for tax compliance.",
        "Is an annual payment option available at a discounted rate?",
        "My billing cycle date is incorrect in the system.",
        "I need an itemized receipt for a corporate expense report.",
        "A regional tax is being applied that does not apply to my country.",
        "I am being charged for user seats that I already removed.",
        "My trial expired and I was billed without any warning email.",
        "The refund has been pending for over three weeks now.",
    ],
    "Technical Issue": [
        "The application crashes every time I attempt to upload a large file.",
        "I am unable to log in and the page loads indefinitely.",
        "Data export fails with a 500 internal server error every time.",
        "The main dashboard is completely blank with no data visible.",
        "Two-factor authentication is not delivering the SMS verification code.",
        "The REST API returns 403 Forbidden even with correct API keys.",
        "The Slack integration stopped functioning after the latest release.",
        "The search feature returns zero results regardless of the query.",
        "The mobile application crashes immediately upon opening.",
        "The password reset email is not arriving in my inbox.",
        "My configured webhook is not triggering on any events.",
        "Real-time push notifications are arriving several hours late.",
        "The delete button for records is completely unresponsive.",
        "Data is not synchronising between the mobile and desktop apps.",
        "I am being automatically signed out every few minutes.",
        "The bulk action tool only processes the first ten records.",
        "Your website is displaying an SSL certificate error in Chrome.",
        "A CSV import fails with a parsing error on a valid, well-formed file.",
        "User access permissions I configured are not being enforced.",
        "Audio call quality is degraded with constant static and dropouts.",
        "Scheduled automated reports are not running at the set time.",
        "The analytics filter panel is displaying incorrect aggregated data.",
        "The print to PDF feature outputs a completely blank document.",
        "The Google Calendar integration is showing incorrect time zones.",
        "I can see another user's private data within my own account.",
    ],
    "Account": [
        "I need to reset my password but have lost access to my old email.",
        "How do I transfer ownership of my account to a new email address?",
        "I want to permanently close my account and erase all my data.",
        "How do I add a new team member to my workspace?",
        "I accidentally removed my own account and need it restored.",
        "My account was deactivated without any explanation or notice.",
        "I need to reassign the organisation administrator to another person.",
        "I created two separate accounts by mistake and need to merge them.",
        "My account appears to have been accessed without my authorisation.",
        "I would like to link my Google account for easier authentication.",
        "The email verification link I received has already expired.",
        "Can I convert my individual account to a business account type?",
        "I need to permanently remove a specific user from my team.",
        "How do I configure role-based access permissions for my team members?",
        "My profile photo is not being saved after I upload a new one.",
        "I am unable to update the company name in the account settings.",
        "How do I export all my account data before I close my account?",
        "Can I create a sub-account for a specific business department?",
        "My session expires even when I select the stay signed in option.",
        "How do I activate two-factor authentication on my account?",
        "I want to register an additional email address on my account.",
        "My display name is not reflecting the changes I saved.",
        "I need to update the phone number associated with my account.",
        "Can login access be restricted to specific approved IP addresses?",
        "My account has been locked after several failed login attempts.",
    ],
    "General Query": [
        "What features differentiate the Professional plan from the Basic plan?",
        "How many business days does a typical refund take to process?",
        "Is there a discounted pricing tier available for nonprofit organisations?",
        "What are the operating hours for your customer support team?",
        "Does your platform comply with GDPR and data protection regulations?",
        "Do you offer a reseller or affiliate partnership programme?",
        "Which programming languages are officially supported by your public API?",
        "What is the guaranteed uptime SLA for enterprise-tier accounts?",
        "Is a native desktop application available for both Mac and Windows?",
        "Do you offer data residency options within the European Economic Area?",
        "Can your solution be deployed on our own on-premise infrastructure?",
        "Which data file formats are supported for importing into the platform?",
        "How does your pricing model scale for organisations with many users?",
        "Do you maintain a publicly accessible product roadmap for upcoming releases?",
        "What is your data retention policy following account cancellation?",
        "Is there an official user community or peer support forum available?",
        "Are dedicated account managers assigned to enterprise-tier customers?",
        "What is the official process for submitting a product feature request?",
        "Which third-party tools and services does your platform integrate with?",
        "Is there an Android version of the mobile application available?",
        "Do you provide a white-label version of your platform for agencies?",
        "What security certifications and compliance standards does your platform hold?",
        "Is professional onboarding or guided setup assistance available?",
        "Can I pause my subscription temporarily without losing my account data?",
        "What is the maximum data storage allowance on each pricing tier?",
    ],
}

# ─── Priority keywords ────────────────────────────────────────────────────────
HIGH_KW   = [
    "crash", "unable to log", "cannot log", "hacked", "fraud", "compromised",
    "suspended", "deactivated", "locked", "charged twice", "not arriving",
    "500 internal", "403 forbidden", "completely blank", "no data visible",
    "crashes immediately", "not delivering", "stolen", "breach", "incorrect amount",
    "double charge", "still charged", "automatically signed out",
]
MEDIUM_KW = [
    "error", "incorrect", "failed", "not functioning", "not synchronising",
    "not saving", "not being saved", "degraded", "not triggering",
    "not enforced", "not running", "not arriving", "no response",
    "not reflecting", "not applying", "not visible", "not working",
    "cannot update", "unable to update",
]

PREFIXES = [
    "", "", "",
    "Hi support team, ", "Hello, ", "Good morning, ",
    "URGENT: ", "Hi there, ", "To the support team, ",
    "Dear support, ", "Greetings, ",
]
SUFFIXES = [
    "", "", "",
    " Please assist.", " Thank you.", " Awaiting your response.",
    " This is very urgent.", " Please help as soon as possible.",
    " Appreciate your help.", " Looking forward to hearing from you.",
]


def assign_priority(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in HIGH_KW):   return "High"
    if any(kw in t for kw in MEDIUM_KW): return "Medium"
    return "Low"


def build_dataset(n_variants: int = 12) -> pd.DataFrame:
    """
    Generate tickets and record the source template_id.
    The template-level 80/20 split prevents data leakage between train and test.
    """
    records = []
    ticket_num = 1000

    for category, template_list in TEMPLATES.items():
        for tmpl_idx, base_text in enumerate(template_list):
            template_id = f"{category[:3].upper()}_T{tmpl_idx:02d}"
            for _ in range(n_variants):
                text = (
                    random.choice(PREFIXES)
                    + base_text
                    + random.choice(SUFFIXES)
                )
                records.append({
                    "ticket_id":   f"TKT-{ticket_num}",
                    "ticket_text": text,
                    "category":    category,
                    "priority":    assign_priority(text),
                    "template_id": template_id,
                })
                ticket_num += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def template_level_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split at the template level so the test set contains
    completely unseen sentence structures (no memorisation).
    """
    train_ids, test_ids = [], []
    for cat in df["category"].unique():
        templates = df[df["category"] == cat]["template_id"].unique().tolist()
        random.shuffle(templates)
        n_test  = max(1, int(len(templates) * test_ratio))
        test_ids  += templates[:n_test]
        train_ids += templates[n_test:]
    train_df = df[df["template_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["template_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

_lemmatizer = WordNetLemmatizer()
_stopwords  = set(stopwords.words("english")) - {"not", "no", "never", "cannot"}


def preprocess(text: str) -> str:
    """Clean and normalise ticket text for TF-IDF vectorisation."""
    text = text.lower()
    text = re.sub(r"http\S+|\S+@\S+", " ", text)          # remove URLs / emails
    text = re.sub(r"[^a-z\s']", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in _stopwords and len(tok) > 2
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODELS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Linear SVC":          LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}


def train_and_evaluate(task_label, Xtr, Xte, y_train, y_test, encoder):
    print(f"\n  {'='*58}")
    print(f"  {task_label} CLASSIFICATION")
    print(f"  {'='*58}")
    print(f"  {'Model':<22} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*58}")

    results = {}
    for name, clf in MODELS.items():
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(Xtr, y_train)
        preds = clf_copy.predict(Xte)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="weighted")
        prec  = precision_score(y_test, preds, average="weighted")
        rec   = recall_score(y_test, preds, average="weighted")
        results[name] = {
            "model": clf_copy, "preds": preds,
            "acc": acc, "f1": f1, "prec": prec, "rec": rec,
        }
        print(f"  {name:<22} {acc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  Best  -> {best}  (F1 = {results[best]['f1']:.4f})")
    return results, best


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SINGLE REPORT IMAGE (Professional Corporate Layout)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_image(
    df, cat_results, best_cat, pri_results, best_pri,
    cat_enc, pri_enc, y_test_cat, y_test_pri, cv_scores
):
    """
    Produce one consolidated, highly professional corporate analysis image.
    """
    model_names  = list(MODELS.keys())
    short_labels = ["LR", "NB", "SVC", "RF"]

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(
        "Automated Support Ticket Classification\nMachine Learning Pipeline Evaluation",
        fontsize=18, fontweight="bold", y=0.98, color=CORP_COLORS["dark"]
    )
    
    # Use gridspec for professional layout
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1, 1.2],
        hspace=0.35, wspace=0.35,
        left=0.06, right=0.96, top=0.88, bottom=0.08,
    )

    ax_cat  = fig.add_subplot(gs[0, 0])
    ax_pri  = fig.add_subplot(gs[0, 1])
    ax_f1   = fig.add_subplot(gs[0, 2])
    ax_cm1  = fig.add_subplot(gs[1, 0])
    ax_cm2  = fig.add_subplot(gs[1, 1])
    ax_cv   = fig.add_subplot(gs[1, 2])

    def format_spines(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DEE2E6")
        ax.spines["bottom"].set_color("#DEE2E6")

    # ── Panel 1: Category distribution (horizontal bar) ───────────────────────
    cat_cnt = df["category"].value_counts().sort_values()
    bars = ax_cat.barh(
        range(len(cat_cnt)), cat_cnt.values,
        color=CORP_COLORS["secondary"], height=0.6, alpha=0.85
    )
    ax_cat.set_yticks(range(len(cat_cnt)))
    ax_cat.set_yticklabels(cat_cnt.index, fontsize=11)
    ax_cat.set_xlabel("Volume of Tickets", fontsize=10)
    ax_cat.set_title("1. Ticket Volume by Category", pad=15)
    for bar, v in zip(bars, cat_cnt.values):
        ax_cat.text(
            v + 3, bar.get_y() + bar.get_height() / 2,
            f"{v:,}", va="center", fontsize=10, fontweight="bold", color=CORP_COLORS["dark"]
        )
    ax_cat.set_xlim(0, cat_cnt.max() * 1.2)
    ax_cat.xaxis.grid(True, linestyle="-", linewidth=0.5, color="#E9ECEF")
    ax_cat.set_axisbelow(True)
    format_spines(ax_cat)

    # ── Panel 2: Priority distribution (vertical bar) ─────────────────────────
    pri_order = ["High", "Medium", "Low"]
    pri_cnt   = [df["priority"].value_counts()[p] for p in pri_order]
    pri_colors = [CORP_COLORS["danger"], CORP_COLORS["warning"], CORP_COLORS["success"]]
    
    bars2 = ax_pri.bar(
        pri_order, pri_cnt,
        color=pri_colors, width=0.55, alpha=0.9
    )
    ax_pri.set_ylabel("Volume of Tickets", fontsize=10)
    ax_pri.set_title("2. Priority Distribution", pad=15)
    for bar, v in zip(bars2, pri_cnt):
        pct = v / len(df) * 100
        ax_pri.text(
            bar.get_x() + bar.get_width() / 2, v + (max(pri_cnt)*0.02),
            f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=CORP_COLORS["dark"]
        )
    ax_pri.set_ylim(0, max(pri_cnt) * 1.25)
    ax_pri.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#E9ECEF")
    ax_pri.set_axisbelow(True)
    format_spines(ax_pri)

    # ── Panel 3: Model F1 comparison (grouped bar) ────────────────────────────
    x = np.arange(len(model_names))
    w = 0.35
    cat_f1 = [cat_results[n]["f1"] for n in model_names]
    pri_f1 = [pri_results[n]["f1"] for n in model_names]

    b1 = ax_f1.bar(
        x - w / 2, cat_f1, w,
        label="Category Prediction", color=CORP_COLORS["primary"], alpha=0.9
    )
    b2 = ax_f1.bar(
        x + w / 2, pri_f1, w,
        label="Priority Prediction", color=CORP_COLORS["info"], alpha=0.9
    )
    for bar, v in list(zip(b1, cat_f1)) + list(zip(b2, pri_f1)):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=CORP_COLORS["dark"]
        )
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(short_labels, fontsize=11)
    ax_f1.set_ylim(0.4, 1.15)
    ax_f1.set_ylabel("F1 Score (Weighted)", fontsize=10)
    ax_f1.set_title("3. Model Performance Comparison", pad=15)
    ax_f1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_f1.axhline(0.90, color=CORP_COLORS["success"], linestyle="--",
                  linewidth=1.5, alpha=0.8, label="0.90 Target")
    ax_f1.text(3.55, 0.91, "Target: 0.90", fontsize=9, color=CORP_COLORS["success"], fontweight="bold")
    ax_f1.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#E9ECEF")
    ax_f1.set_axisbelow(True)
    format_spines(ax_f1)

    # ── Helper for Confusion Matrices ─────────────────────────────────────────
    def plot_cm(ax, cm, labels, title, cmap):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=1, linecolor="#FFFFFF",
            xticklabels=[c.replace(" ", "\n") for c in labels],
            yticklabels=[c.replace(" ", "\n") for c in labels]
        )
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
        ax.set_ylabel("True Label", fontsize=10, labelpad=10)
        ax.set_title(title, pad=15)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # ── Panel 4: Confusion matrix — Category ──────────────────────────────────
    cm_cat = confusion_matrix(y_test_cat, cat_results[best_cat]["preds"])
    plot_cm(ax_cm1, cm_cat, cat_enc.classes_, f"4. Category Confusion Matrix ({best_cat})", "Blues")

    # ── Panel 5: Confusion matrix — Priority ──────────────────────────────────
    cm_pri = confusion_matrix(y_test_pri, pri_results[best_pri]["preds"])
    plot_cm(ax_cm2, cm_pri, pri_enc.classes_, f"5. Priority Confusion Matrix ({best_pri})", "Teal" if "Teal" in plt.colormaps() else "GnBu")

    # ── Panel 6: Cross-validation ─────────────────────────────────────────────
    folds = np.arange(1, len(cv_scores) + 1)
    ax_cv.fill_between(
        folds,
        cv_scores.mean() - cv_scores.std(),
        cv_scores.mean() + cv_scores.std(),
        alpha=0.15, color=CORP_COLORS["primary"], label="± 1 Standard Deviation"
    )
    ax_cv.plot(
        folds, cv_scores, "o-",
        color=CORP_COLORS["primary"], linewidth=2.5, markersize=10,
        markerfacecolor="#FFFFFF", markeredgewidth=2.5
    )
    ax_cv.axhline(
        cv_scores.mean(), color=CORP_COLORS["dark"], linestyle="--",
        linewidth=2, label=f"Mean F1 = {cv_scores.mean():.4f}"
    )
    for i, s in zip(folds, cv_scores):
        ax_cv.text(i, s + (cv_scores.max()-cv_scores.min())*0.05 + 0.005, f"{s:.4f}",
                   ha="center", fontsize=10, fontweight="bold", color=CORP_COLORS["primary"])
    ax_cv.set_xticks(list(folds))
    ax_cv.set_xticklabels([f"Fold {i}" for i in folds], fontsize=11)
    
    ymin = min(0.8, cv_scores.min() - 0.05)
    ax_cv.set_ylim(ymin, 1.05)
    ax_cv.set_ylabel("F1 Score", fontsize=10)
    ax_cv.set_title(f"6. Stability: 5-Fold Cross-Validation ({best_cat})", pad=15)
    ax_cv.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_cv.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#E9ECEF")
    ax_cv.set_axisbelow(True)
    format_spines(ax_cv)

    # ── Save single report image ───────────────────────────────────────────────
    out_path = "outputs/analysis_report.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  Saved  ->  {out_path}  ({size_kb} KB)")


def chart_eda(df):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Exploratory Data Analysis — Support Ticket Dataset",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # 1. Category count
    cat_cnt = df["category"].value_counts()
    bars = ax1.bar(range(len(cat_cnt)), cat_cnt.values,
                   color=[CAT_COLORS[c] for c in cat_cnt.index], width=0.6)
    ax1.set_xticks(range(len(cat_cnt)))
    ax1.set_xticklabels([c.replace(" ", "\n") for c in cat_cnt.index], fontsize=9)
    ax1.set_title("Ticket Count by Category")
    ax1.set_ylabel("Count")
    for bar, v in zip(bars, cat_cnt.values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 5, str(v),
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color="#C9D1D9")
    ax1.set_ylim(0, cat_cnt.max() * 1.2)

    # 2. Priority donut
    pri_cnt = df["priority"].value_counts()[["High", "Medium", "Low"]]
    wedge, _, auto = ax2.pie(
        pri_cnt.values, labels=pri_cnt.index, autopct="%1.1f%%",
        colors=[PRI_COLORS[p] for p in pri_cnt.index],
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "#0D1117", "linewidth": 2},
        textprops={"color": "#C9D1D9", "fontsize": 9},
    )
    for a in auto:
        a.set_fontweight("bold")
    ax2.set_title("Priority Distribution")

    # 3. Stacked bar — priority by category
    cp = df.groupby(["category", "priority"]).size().unstack(fill_value=0)
    cp = cp[[p for p in ["High", "Medium", "Low"] if p in cp.columns]]
    bottom = np.zeros(len(cp))
    for pri in cp.columns:
        ax3.bar(range(len(cp)), cp[pri].values, bottom=bottom,
                label=pri, color=PRI_COLORS[pri], width=0.6)
        bottom += cp[pri].values
    ax3.set_xticks(range(len(cp)))
    ax3.set_xticklabels([c.replace(" ", "\n") for c in cp.index], fontsize=9)
    ax3.set_title("Priority Stack by Category")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8, title="Priority", title_fontsize=8)

    # 4. Word count histogram
    for cat in df["category"].unique():
        ax4.hist(df[df["category"] == cat]["word_count"],
                 bins=15, alpha=0.65, label=cat, color=CAT_COLORS[cat])
    ax4.set_title("Word Count Distribution")
    ax4.set_xlabel("Words per Ticket")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=7)

    # 5. Avg word count by category
    avg_wc = df.groupby("category")["word_count"].mean().sort_values()
    ax5.barh(range(len(avg_wc)), avg_wc.values,
             color=[CAT_COLORS[c] for c in avg_wc.index], height=0.6)
    ax5.set_yticks(range(len(avg_wc)))
    ax5.set_yticklabels([c.replace(" ", "\n") for c in avg_wc.index], fontsize=9)
    ax5.set_title("Avg Word Count by Category")
    ax5.set_xlabel("Avg Words")
    for i, v in enumerate(avg_wc.values):
        ax5.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9,
                 fontweight="bold", color="#C9D1D9")

    # 6. Text length boxplot by priority
    data_bp = [df[df["priority"] == p]["text_length"].values
               for p in ["High", "Medium", "Low"]]
    bp = ax6.boxplot(data_bp, tick_labels=["High", "Medium", "Low"],
                     patch_artist=True,
                     medianprops={"color": "#0D1117", "linewidth": 2},
                     whiskerprops={"color": "#8B949E"},
                     capprops={"color": "#8B949E"},
                     flierprops={"marker": "o", "markersize": 3,
                                 "markerfacecolor": "#8B949E", "alpha": 0.5})
    for patch, col in zip(bp["boxes"], [PRI_COLORS[p] for p in ["High", "Medium", "Low"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax6.set_title("Text Length by Priority")
    ax6.set_ylabel("Characters")

    _save(fig, "01_eda.png")


def chart_wordclouds(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Word Clouds — Most Frequent Terms per Category",
                 fontsize=15, fontweight="bold")
    sw = set(stopwords.words("english"))
    cmaps = {
        "Billing":        "Blues",
        "Technical Issue":"Reds",
        "Account":        "Greens",
        "General Query":  "YlOrBr",
    }
    for ax, cat in zip(axes.flat, TEMPLATES.keys()):
        corpus = " ".join(df[df["category"] == cat]["ticket_text"])
        wc = WordCloud(
            width=700, height=350,
            background_color="#161B22",
            colormap=cmaps[cat],
            stopwords=sw,
            max_words=70,
            contour_width=0,
            prefer_horizontal=0.9,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cat, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
    _save(fig, "02_wordclouds.png")


def chart_model_comparison(cat_results, pri_results):
    model_names  = list(MODELS.keys())
    short_labels = ["LR", "NB", "SVC", "RF"]
    metric_keys  = ["acc", "f1", "prec", "rec"]
    metric_label = ["Accuracy", "F1 Score", "Precision", "Recall"]
    metric_cols  = [PALETTE["blue"], PALETTE["green"],
                    PALETTE["yellow"], PALETTE["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Performance Comparison — All Classifiers",
                 fontsize=15, fontweight="bold")

    for ax, results, title in [
        (axes[0], cat_results, "Category Classification"),
        (axes[1], pri_results, "Priority Classification"),
    ]:
        x = np.arange(len(model_names))
        w = 0.18
        for i, (key, label, col) in enumerate(
                zip(metric_keys, metric_label, metric_cols)):
            vals = [results[n][key] for n in model_names]
            bars = ax.bar(x + i * w, vals, w, label=label,
                          color=col, alpha=0.9, edgecolor="#0D1117", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color="#C9D1D9")

        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_ylim(0.4, 1.18)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(0.90, color=PALETTE["red"],
                   linestyle="--", linewidth=1, alpha=0.5)
        ax.text(3.75, 0.905, "0.90", color=PALETTE["red"], fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    _save(fig, "03_model_comparison.png")


def chart_confusion_matrices(cat_results, best_cat, pri_results, best_pri,
                              cat_enc, pri_enc, yc_te, yp_te):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrices — Best Models", fontsize=15, fontweight="bold")

    for ax, preds, enc, cmap, title in [
        (axes[0], cat_results[best_cat]["preds"], cat_enc,
         "Blues", f"Category  |  {best_cat}"),
        (axes[1], pri_results[best_pri]["preds"], pri_enc,
         "Greens", f"Priority  |  {best_pri}"),
    ]:
        actual = yc_te if enc is cat_enc else yp_te
        cm = confusion_matrix(actual, preds)
        im = ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(enc.classes_)))
        ax.set_yticks(range(len(enc.classes_)))
        ax.set_xticklabels(enc.classes_, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(enc.classes_, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title)
        thresh = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="#0D1117" if cm[r, c] > thresh else "#C9D1D9")
        plt.colorbar(im, ax=ax, shrink=0.8)

    _save(fig, "04_confusion_matrices.png")


def chart_cross_validation(cv_scores, model_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    folds = range(1, len(cv_scores) + 1)

    ax.fill_between(folds,
                    cv_scores.mean() - cv_scores.std(),
                    cv_scores.mean() + cv_scores.std(),
                    alpha=0.15, color=PALETTE["blue"], label="± 1 Std Dev")
    ax.plot(folds, cv_scores, "o-",
            color=PALETTE["blue"], linewidth=2.5,
            markersize=10, markerfacecolor="#0D1117",
            markeredgewidth=2, label="Fold F1")
    ax.axhline(cv_scores.mean(), color=PALETTE["yellow"],
               linestyle="--", linewidth=2,
               label=f"Mean F1 = {cv_scores.mean():.4f}")

    for i, s in zip(folds, cv_scores):
        ax.text(i, s + 0.006, f"{s:.4f}",
                ha="center", fontsize=9, fontweight="bold", color="#C9D1D9")

    ax.set_xticks(list(folds))
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylim(max(0.5, cv_scores.min() - 0.1), 1.05)
    ax.set_xlabel("Cross-Validation Fold", fontsize=10)
    ax.set_ylabel("F1 Score (Weighted)", fontsize=10)
    ax.set_title(f"5-Fold Stratified Cross-Validation  |  {model_name}")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    _save(fig, "05_cross_validation.png")


def chart_kpi_dashboard(cat_results, best_cat, pri_results, best_pri, cv_scores):
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Business Intelligence Dashboard — Support Ticket ML System",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

    kpi_ax  = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bar_ax  = fig.add_subplot(gs[1, :])

    # KPI cards
    kpis = [
        (f"{cat_results[best_cat]['acc']*100:.1f}%",
         "Category Accuracy", f"Best: {best_cat}", PALETTE["blue"]),
        (f"{pri_results[best_pri]['acc']*100:.1f}%",
         "Priority Accuracy", f"Best: {best_pri}", PALETTE["green"]),
        (f"{cat_results[best_cat]['f1']:.4f}",
         "Category F1 Score", "Weighted", PALETTE["yellow"]),
        (f"{cv_scores.mean():.4f}",
         "Cross-Val Mean F1", f"Std: {cv_scores.std():.4f}", PALETTE["purple"]),
    ]
    for ax, (val, title, sub, col) in zip(kpi_ax, kpis):
        ax.set_facecolor("#1C2128")
        for sp in ax.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(1.5)
        ax.text(0.5, 0.60, val, ha="center", va="center",
                fontsize=34, color=col, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.28, title, ha="center", va="center",
                fontsize=11, color="#C9D1D9", fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.10, sub, ha="center", va="center",
                fontsize=8, color="#8B949E", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # F1 comparison bar
    bar_ax.set_facecolor("#161B22")
    for sp in bar_ax.spines.values():
        sp.set_edgecolor("#30363D")
    model_names = list(MODELS.keys())
    x = np.arange(len(model_names))
    w = 0.32
    for offset, results, label, col in [
        (-w/2, cat_results, "Category F1", PALETTE["blue"]),
        ( w/2, pri_results, "Priority F1",  PALETTE["green"]),
    ]:
        vals = [results[n]["f1"] for n in model_names]
        bars = bar_ax.bar(x + offset, vals, w, label=label,
                          color=col, alpha=0.9,
                          edgecolor="#0D1117", linewidth=0.5)
        for bar, v in zip(bars, vals):
            bar_ax.text(bar.get_x() + bar.get_width()/2,
                        v + 0.006, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#C9D1D9")

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(
        ["Logistic\nRegression", "Naive\nBayes", "Linear\nSVC", "Random\nForest"],
        fontsize=10, color="#C9D1D9")
    bar_ax.set_ylim(0.3, 1.15)
    bar_ax.set_ylabel("Weighted F1 Score", fontsize=10)
    bar_ax.set_title("F1 Score — Category vs Priority Across All Models", fontsize=12)
    bar_ax.legend(fontsize=9)
    bar_ax.axhline(0.90, color=PALETTE["red"],
                   linestyle="--", linewidth=1.2, alpha=0.6)
    bar_ax.text(3.6, 0.91, "0.90 target",
                color=PALETTE["red"], fontsize=9)
    bar_ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    bar_ax.set_axisbelow(True)

    _save(fig, "06_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 62

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 1 — Dataset Generation")
    print(sep)

    df = build_dataset(n_variants=12)
    df["word_count"]   = df["ticket_text"].apply(lambda x: len(x.split()))
    df["text_length"]  = df["ticket_text"].apply(len)
    df["clean_text"]   = df["ticket_text"].apply(preprocess)

    train_df, test_df = template_level_split(df, test_ratio=0.20)

    print(f"  Total tickets       : {len(df):,}")
    print(f"  Train set           : {len(train_df):,}  (80% of templates)")
    print(f"  Test set            : {len(test_df):,}   (20% of templates — unseen)")
    print(f"  Avg words/ticket    : {df['word_count'].mean():.1f}")
    print()
    print(f"  {'Category':<20} {'Count':>6}  {'Share':>6}")
    print(f"  {'-'*36}")
    for cat, cnt in df["category"].value_counts().items():
        print(f"  {cat:<20} {cnt:>6}  {cnt/len(df)*100:>5.1f}%")
    print()
    print(f"  {'Priority':<12} {'Count':>6}  {'Share':>6}")
    print(f"  {'-'*28}")
    for pri in ["High", "Medium", "Low"]:
        cnt = (df["priority"] == pri).sum()
        print(f"  {pri:<12} {cnt:>6}  {cnt/len(df)*100:>5.1f}%")

    # ── Step 2: Features ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 2 — TF-IDF Feature Engineering")
    print(sep)

    cat_enc = LabelEncoder()
    pri_enc = LabelEncoder()

    y_train_cat = cat_enc.fit_transform(train_df["category"])
    y_test_cat  = cat_enc.transform(test_df["category"])
    y_train_pri = pri_enc.fit_transform(train_df["priority"])
    y_test_pri  = pri_enc.transform(test_df["priority"])

    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    Xtr = tfidf.fit_transform(train_df["clean_text"])
    Xte = tfidf.transform(test_df["clean_text"])

    print(f"  Train matrix shape  : {Xtr.shape[0]:,} x {Xtr.shape[1]:,}")
    print(f"  Test  matrix shape  : {Xte.shape[0]:,} x {Xte.shape[1]:,}")
    print(f"  Vocabulary size     : {len(tfidf.vocabulary_):,} terms")
    print(f"  Matrix sparsity     : {(1 - Xtr.nnz/(Xtr.shape[0]*Xtr.shape[1]))*100:.1f}%")

    # ── Step 3: Train & Evaluate ──────────────────────────────────────────────
    cat_results, best_cat = train_and_evaluate(
        "CATEGORY", Xtr, Xte, y_train_cat, y_test_cat, cat_enc)
    pri_results, best_pri = train_and_evaluate(
        "PRIORITY",  Xtr, Xte, y_train_pri, y_test_pri, pri_enc)

    # ── Step 4: Classification Reports ───────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  DETAILED REPORT — {best_cat}  (Category)")
    print(f"  {sep}")
    print(classification_report(
        y_test_cat, cat_results[best_cat]["preds"],
        target_names=cat_enc.classes_, digits=4))

    print(f"  {sep}")
    print(f"  DETAILED REPORT — {best_pri}  (Priority)")
    print(f"  {sep}")
    print(classification_report(
        y_test_pri, pri_results[best_pri]["preds"],
        target_names=pri_enc.classes_, digits=4))

    # ── Step 5: Cross-Validation ──────────────────────────────────────────────
    print(f"  {sep}")
    print(f"  5-FOLD CROSS-VALIDATION — {best_cat}")
    print(f"  {sep}")
    X_all = tfidf.transform(df["clean_text"])
    y_all = cat_enc.transform(df["category"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cat_results[best_cat]["model"], X_all, y_all,
        cv=cv, scoring="f1_weighted")

    for i, s in enumerate(cv_scores, 1):
        print(f"  Fold {i}  :  {s:.4f}")
    print(f"  Mean   :  {cv_scores.mean():.4f}  +/-  {cv_scores.std():.4f}")

    # ── Step 6: Live Predictions ───────────────────────────────────────────────
    def predict(text):
        vec = tfidf.transform([preprocess(text)])
        cat = cat_enc.inverse_transform(
            cat_results[best_cat]["model"].predict(vec))[0]
        pri = pri_enc.inverse_transform(
            pri_results[best_pri]["model"].predict(vec))[0]
        return cat, pri

    print(f"\n  {sep}")
    print("  LIVE PREDICTION DEMO")
    print(f"  {sep}")
    demo_tickets = [
        "My account was hacked and I cannot log in at all.",
        "Please send me an invoice for this month's payment.",
        "The dashboard is completely blank after the latest update.",
        "What is the difference between your Basic and Pro plans?",
        "I was charged but I am supposed to be on the free trial.",
        "How do I add a new user to my team workspace?",
        "The mobile app crashes every time I open it on my iPhone.",
        "Do you offer any discounts for educational institutions?",
    ]
    CAT_ICONS = {"Billing": "[BILLING]", "Technical Issue": "[TECHNICAL]",
                 "Account": "[ACCOUNT]", "General Query": "[GENERAL]"}
    PRI_ICONS = {"High": "[HIGH]", "Medium": "[MED]", "Low": "[LOW]"}
    for ticket in demo_tickets:
        cat, pri = predict(ticket)
        display = (ticket[:70] + "...") if len(ticket) > 70 else ticket
        print(f"\n  Ticket  : \"{display}\"")
        print(f"  Result  : {CAT_ICONS[cat]}  {PRI_ICONS[pri]}")

    # ── Step 7: Single Report Image ────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  GENERATING REPORT IMAGE")
    print(f"  {sep}")
    generate_report_image(
        df=df,
        cat_results=cat_results, best_cat=best_cat,
        pri_results=pri_results, best_pri=best_pri,
        cat_enc=cat_enc, pri_enc=pri_enc,
        y_test_cat=y_test_cat, y_test_pri=y_test_pri,
        cv_scores=cv_scores,
    )

    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL SUMMARY")
    print(sep)
    print(f"  Dataset            : {len(df):,} tickets | 4 categories | 3 priorities")
    print(f"  Evaluation method  : Template-level split (no data leakage)")
    print(f"  Train / Test       : {len(train_df):,} / {len(test_df):,}")
    print(f"  TF-IDF vocabulary  : {len(tfidf.vocabulary_):,} terms  |  n-grams (1,2)")
    print()
    print(f"  CATEGORY TASK")
    print(f"  {'-'*50}")
    for n in MODELS:
        r = cat_results[n]
        flag = "  <- BEST" if n == best_cat else ""
        print(f"  {n:<22}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}{flag}")
    print()
    print(f"  PRIORITY TASK")
    print(f"  {'-'*50}")
    for n in MODELS:
        r = pri_results[n]
        flag = "  <- BEST" if n == best_pri else ""
        print(f"  {n:<22}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}{flag}")
    print()
    print(f"  Cross-Val F1 (5-fold) : {cv_scores.mean():.4f}  +/-  {cv_scores.std():.4f}")
    print()
    out = "outputs/analysis_report.png"
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"  Output image  :  {out}  ({size//1024} KB)")
    print(f"\n{sep}")
    print("  PIPELINE COMPLETE")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
