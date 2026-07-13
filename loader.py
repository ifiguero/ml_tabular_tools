from __future__ import annotations

from pathlib import Path
from typing import Callable

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import re



class BasicLoader:
    def __init__(
        self,
        file: str | Path,
        target: list[str],
        continuous: list[str],
        discrete: list[str],
        labels: dict[str, str] | None = None,
        title_map: dict[str, str] | None = None,
        csv_separator: str=',',
    ):
        self.file = Path(file)

        self.targets = list(target)
        self.continuous = list(continuous)
        self.discrete = list(discrete)

        self.labels = {}
        self.clean = {}
        for item in self.continuous:
            self.labels[item] = item
            self.clean[item] = re.sub(r'[^\w\-]', '', str(item).strip().replace(' ', '_'))

        for item in self.discrete:
            self.labels[item] = item
            self.clean[item] = re.sub(r'[^\w\-]', '', str(item).strip().replace(' ', '_'))

        for item in self.targets:
            self.labels[item] = item
            self.clean[item] = re.sub(r'[^\w\-]', '', str(item).strip().replace(' ', '_'))

        self.datasets: dict[str, pd.DataFrame] = {}
        self.state: dict[str, pd.DataFrame] = {}

        self._load(separator=csv_separator)
        self._validate_variables()
        self.build_datasets()
        self._validate_targets()

    ###########################################################################
    # Loading
    ###########################################################################

    def _load(self, separator: str=','):

        suffix = self.file.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.file, sep=separator)

        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(self.file)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self.state["raw"] = df


    def _validate_targets(self):

        for target in self.targets:

            dfi = self.datasets[target]
            groups = sorted(dfi[target].dropna().unique())

            if len(groups) != 2:
                raise ValueError(
                    f"Target '{target}' must have exactly two unique values"
                )


    def _validate_variables(self):

        required = (
            self.continuous +
            self.discrete
        )

        missing = [c for c in required if c not in self.state["raw"].columns]

        if missing:
            raise ValueError(
                f"Missing required variable column: {missing}\n Available Columns: {self.state['raw'].columns}"
            )

    ###########################################################################
    # Dataset creation
    ###########################################################################

    def build_datasets(self):
        features = (
            self.continuous +
            self.discrete
        )

        for target in self.targets:

            cols = features + [target]

            self.datasets[target] = (
                self.state["raw"].loc[:, cols]
                .copy()
            )

    ###########################################################################
    # Access
    ###########################################################################
    def set_label(self, target: str, label: str):
        self.labels[target] = label

    def __getitem__(self, target: str):
        return self.datasets[target]


    def keys(self):
        return self.datasets.keys()

    ###########################################################################
    # Export
    ###########################################################################

    def get_dataset(
        self,
        target: str,
        remap_smaller_is_zero: bool = False,
    ) -> pd.DataFrame:

        dfi = self.datasets[target].copy()

        if not remap_smaller_is_zero:
            return dfi

        groups = sorted(dfi[target].dropna().unique())

        mapping = {
            groups[0]: 0,
            groups[1]: 1,
        }

        dfi[target] = dfi[target].map(mapping)

        return dfi

    def save_datasets(
        self,
        outfile: str | Path,
    ):

        outfile = Path(outfile)

        outfile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with pd.ExcelWriter(outfile) as writer:

            for name, df in self.datasets.items():

                df.to_excel(
                    writer,
                    sheet_name=name[:31],
                    index=False,
                )

    ###########################################################################
    # Univariate analysis
    ###########################################################################

    def _univariate_analysis(self, dfi: pd.DataFrame, target: str) -> pd.DataFrame:

        groups = sorted(dfi[target].dropna().unique())

        if len(groups) != 2:
            raise ValueError(
                f"Target '{target}' must have exactly two unique values"
            )

        group1, group2 = groups

        glabel = {
            group1: str(group1),
            group2: str(group2),
        }

        data_group1 = dfi[dfi[target] == group1]
        data_group2 = dfi[dfi[target] == group2]

        results = []

        #######################################################################
        # Continuous variables
        #######################################################################

        for var in self.continuous:

            x = dfi[var].dropna()
            g1 = data_group1[var].dropna()
            g2 = data_group2[var].dropna()

            if len(x) >= 3:
                _, p_normal = shapiro(x)
            else:
                p_normal = 0.0

            normal = p_normal >= 0.05

            if normal:

                mean_total = x.mean()
                std_total = x.std()

                mean1 = g1.mean()
                std1 = g1.std()

                mean2 = g2.mean()
                std2 = g2.std()

                _, pvalue = ttest_ind(
                    g1,
                    g2,
                    equal_var=False,
                    nan_policy="omit",
                )

                total = f"{mean_total:.1f} ± {std_total:.1f}"
                value1 = f"{mean1:.1f} ± {std1:.1f}"
                value2 = f"{mean2:.1f} ± {std2:.1f}"

            else:

                median_total = x.median()
                q1_total = x.quantile(0.25)
                q3_total = x.quantile(0.75)

                median1 = g1.median()
                q1_1 = g1.quantile(0.25)
                q3_1 = g1.quantile(0.75)

                median2 = g2.median()
                q1_2 = g2.quantile(0.25)
                q3_2 = g2.quantile(0.75)

                _, pvalue = mannwhitneyu(g1, g2)

                total = f"{median_total:.1f} ({q1_total:.1f} - {q3_total:.1f})"
                value1 = f"{median1:.1f} ({q1_1:.1f} - {q3_1:.1f})"
                value2 = f"{median2:.1f} ({q1_2:.1f} - {q3_2:.1f})"

            results.append(
                [
                    self.labels.get(var, var),
                    "",
                    value1,
                    value2,
                    f"{pvalue:.3f}",
                    "*" if pvalue < 0.05 else "NS",
                    total,
                ]
            )

        #######################################################################
        # Discrete variables
        #######################################################################

        for var in self.discrete:

            freq = (
                dfi.groupby([target, var])
                .size()
                .unstack(fill_value=0)
            )

            perc = freq.div(freq.sum(axis=1), axis=0) * 100

            if freq.shape[1] == 2:
                _, pvalue = fisher_exact(freq.values)
                positive_col = freq.columns[1]
            else:
                _, pvalue, _, _ = chi2_contingency(freq)
                positive_col = freq.columns[-1]

            n1 = freq.loc[group1, positive_col]
            n2 = freq.loc[group2, positive_col]

            p1 = perc.loc[group1, positive_col]
            p2 = perc.loc[group2, positive_col]

            total_n = n1 + n2
            total_p = total_n / len(dfi) * 100

            results.append(
                [
                    self.labels.get(var, var),
                    "",
                    f"{p1:.1f} ({n1}/{len(data_group1)})",
                    f"{p2:.1f} ({n2}/{len(data_group2)})",
                    f"{pvalue:.3f}",
                    "*" if pvalue < 0.05 else "NS",
                    f"{total_p:.1f} ({total_n}/{len(dfi)})",
                ]
            )

        return pd.DataFrame(
            results,
            columns=[
                "Variable",
                "Unidad",
                f"{glabel[group1]}\n n={len(data_group1)}",
                f"{glabel[group2]}\n n={len(data_group2)}",
                "Pvalue",
                " ",
                "Total",
            ],
        )


    def univariate_analysis(self, target: str) -> pd.DataFrame:
        return self._univariate_analysis(
            self.datasets[target],
            target,
        )


    def save_univariate_analysis(
        self,
        outfile: str | Path,
    ):

        outfile = Path(outfile)

        outfile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with pd.ExcelWriter(outfile) as writer:

            for target, dfi in self.datasets.items():

                result = self._univariate_analysis(
                    dfi,
                    target,
                )

                result.to_excel(
                    writer,
                    sheet_name=target[:31],
                    index=False,
                )

    ###########################################################################
    # PCA
    ###########################################################################

    def plot_pca(
        self,
        target: str,
        outfile: str | Path | None = None,
        n_components: int = 2,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 300,
        scale: bool = True,
        arrow_scale: float = 2.0,
        arrow_color: str = "green",
        arrow_alpha: float = 0.7,
        arrow_head_width: float = 0.05,
        point_size: float = 50,
        point_alpha: float = 1.0,
        colors: dict | None = None,
        class_labels: dict | None = None,
        title: str | None = None,
        xlabel: str = "Principal Component 1",
        ylabel: str = "Principal Component 2",
        feature_labels: dict[str, str] | None = None,
        grid: bool = True,
    ):


        if target not in self.datasets:
            raise KeyError(f"Unknown target '{target}'")

        dfi = self.datasets[target].dropna()

        X = dfi[self.continuous]
        y = dfi[target].reset_index(drop=True)

        if scale:
            Xp = StandardScaler().fit_transform(X)
        else:
            Xp = X.to_numpy()

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(Xp)

        principal_df = pd.DataFrame(
            pcs,
            columns=[
                f"Principal Component {i + 1}"
                for i in range(n_components)
            ],
        )

        final_df = pd.concat(
            [principal_df, y],
            axis=1,
        )

        loadings = pca.components_.T

        if feature_labels is None:
            feature_labels = {
                c: self.labels.get(c, c)
                for c in self.continuous
            }

        groups = sorted(final_df[target].unique())

        if colors is None:
            default_colors = [
                "tab:blue",
                "tab:red",
                "tab:green",
                "tab:orange",
                "tab:purple",
                "tab:brown",
            ]
            colors = {
                g: default_colors[i % len(default_colors)]
                for i, g in enumerate(groups)
            }

        if class_labels is None:
            class_labels = {
                g: str(g)
                for g in groups
            }

        if title is None:
            title = f"PCA - {self.labels[target]}"

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        for group in groups:

            idx = final_df[target] == group

            ax.scatter(
                final_df.loc[idx, "Principal Component 1"],
                final_df.loc[idx, "Principal Component 2"],
                c=colors[group],
                s=point_size,
                alpha=point_alpha,
                label=class_labels[group],
            )

        for i, var in enumerate(self.continuous):

            ax.arrow(
                0,
                0,
                loadings[i, 0] * arrow_scale,
                loadings[i, 1] * arrow_scale,
                color=arrow_color,
                alpha=arrow_alpha,
                head_width=arrow_head_width,
            )

            ax.text(
                loadings[i, 0] * arrow_scale * 1.05,
                loadings[i, 1] * arrow_scale * 1.05,
                feature_labels.get(var, var),
                ha="center",
                va="center",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        if grid:
            ax.grid(True)

        if outfile is None:
            outdir = Path("img")
            outdir.mkdir(
                parents=True,
                exist_ok=True,
            )
            outfile = outdir / f"{self.clean[target]}_PCA.png"
        else:
            outfile = Path(outfile)
            outfile.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        fig.tight_layout()
        fig.savefig(outfile)

        plt.close(fig)

        return {
            "pca": pca,
            "loadings": loadings,
            "scores": final_df,
            "explained_variance": pca.explained_variance_ratio_,
            "outfile": outfile,
        }


    def plot_all_pca(
        self,
        **kwargs,
    ):

        results = {}

        for target in self.datasets:

            results[target] = self.plot_pca(
                target,
                **kwargs,
            )

        return results

    ###########################################################################
    # Heatmap
    ###########################################################################

    def plot_heatmap(
        self,
        target: str,
        outfile: str | Path | None = None,
        method: str = "pearson",
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 300,
        cmap: str = "coolwarm",
        annot: bool = True,
        fmt: str = ".2f",
        linewidths: float = 0.5,
        linecolor: str = "white",
        square: bool = False,
        cbar: bool = True,
        vmin: float = -1.0,
        vmax: float = 1.0,
        title: str | None = None,
        feature_labels: dict[str, str] | None = None,
        include_target: bool = True,
        rotation_x: float = 45,
        rotation_y: float = 0,
    ):


        if target not in self.datasets:
            raise KeyError(f"Unknown target '{target}'")

        dfi = self.datasets[target].copy()

        if not include_target:
            dfi = dfi.drop(columns=[target])

        if feature_labels is None:
            feature_labels = {
                c: self.labels.get(c, c)
                for c in dfi.columns
            }

        dfi = dfi.rename(columns=feature_labels)

        corr = dfi.corr(method=method, numeric_only=True)

        if title is None:
            title = f"Correlation Heatmap - {self.labels[target]}"

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        sns.heatmap(
            corr,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            linewidths=linewidths,
            linecolor=linecolor,
            square=square,
            cbar=cbar,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=rotation_x)
        ax.tick_params(axis="y", rotation=rotation_y)

        if outfile is None:
            outdir = Path("img")
            outdir.mkdir(
                parents=True,
                exist_ok=True,
            )
            outfile = outdir / f"{self.clean[target]}_Heatmap.png"
        else:
            outfile = Path(outfile)
            outfile.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        fig.tight_layout()
        fig.savefig(outfile)

        plt.close(fig)

        return {
            "correlation": corr,
            "outfile": outfile,
        }

    def plot_all_heatmaps(
        self,
        **kwargs,
    ):

        results = {}

        for target in self.datasets:

            results[target] = self.plot_heatmap(
                target,
                **kwargs,
            )

        return results

    ###########################################################################
    # Distributions
    ###########################################################################

    def plot_distribution(
        self,
        target: str,
        variable: str,
        kind: str = "hist",
        outfile: str | Path | None = None,
        figsize: tuple[float, float] = (12, 6),
        dpi: int = 300,
        xlabel: str | None = None,
        ylabel: str = "Frequency",
        title: str | None = None,
        feature_labels: dict[str, str] | None = None,
        class_labels: dict | None = None,
        xticks: list[str] | None = None,
        palette=None,
        kde: bool = True,
        bins="auto",
        stat: str = "count",
        common_norm: bool = False,
        alpha: float = 0.6,
        legend_title: str | None = None,
        legend_loc_hist: str = "upper right",
        legend_loc_count: str = "upper left",
        show: bool = False,
    ):


        if target not in self.datasets:
            raise KeyError(f"Unknown target '{target}'")

        if variable not in self.continuous + self.discrete:
            raise KeyError(f"Unknown variable '{variable}'")

        dfi = self.datasets[target].copy()

        if feature_labels is None:
            feature_labels = {
                c: self.labels.get(c, c)
                for c in dfi.columns
            }

        groups = sorted(dfi[target].dropna().unique())

        if class_labels is None:
            class_labels = {
                g: str(g)
                for g in groups
            }

        if legend_title is None:
            legend_title = feature_labels.get(target, target)

        if xlabel is None:
            xlabel = feature_labels.get(variable, variable)

        if title is None:
            title = feature_labels.get(variable, variable)

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        if kind.lower() == "hist":

            sns.histplot(
                data=dfi,
                x=variable,
                hue=target,
                kde=kde,
                bins=bins,
                stat=stat,
                common_norm=common_norm,
                alpha=alpha,
                palette=palette,
                ax=ax,
            )

            handles, _ = ax.get_legend_handles_labels()

            ax.legend(
                handles,
                [class_labels[g] for g in groups],
                title=legend_title,
                loc=legend_loc_hist,
            )

            suffix = "hist"

        elif kind.lower() == "count":

            sns.countplot(
                data=dfi,
                x=variable,
                hue=target,
                palette=palette,
                ax=ax,
            )

            if xticks is not None:
                ax.set_xticks(
                    range(len(xticks)),
                    xticks,
                )

            handles, _ = ax.get_legend_handles_labels()

            ax.legend(
                handles,
                [class_labels[g] for g in groups],
                title=legend_title,
                loc=legend_loc_count,
            )

            suffix = "dist"

        else:
            raise ValueError("kind must be either 'hist' or 'count'")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        if outfile is None:
            outdir = Path("img")
            outdir.mkdir(
                parents=True,
                exist_ok=True,
            )
            outfile = outdir / f"{self.clean[target]}_{suffix}_{self.clean[variable]}.png"
        else:
            outfile = Path(outfile)
            outfile.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        fig.tight_layout()
        fig.savefig(outfile)

        plt.close(fig)

        return 0


    def plot_all_distributions(
        self,
        hist_variables: list[str] | None = None,
        count_variables: list[str] | None = None,
        **kwargs,
    ):

        if hist_variables is None:
            hist_variables = list(self.continuous)

        if count_variables is None:
            count_variables = list(self.discrete)

        results = {}

        for target in self.datasets:

            for var in hist_variables:
                self.plot_distribution(
                    target=target,
                    variable=var,
                    kind="hist",
                    **kwargs,
                )

            for var in count_variables:
                self.plot_distribution(
                    target=target,
                    variable=var,
                    kind="count",
                    **kwargs,
                )

        return 0

    ###########################################################################
    # Boxplot
    ###########################################################################

    def plot_boxplot(
        self,
        target: str,
        variable: str,
        outfile: str | Path | None = None,
        figsize: tuple[float, float] = (12, 6),
        dpi: int = 300,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        feature_labels: dict[str, str] | None = None,
        class_labels: dict | None = None,
        palette=None,
        linewidth: float = 1.5,
        width: float = 0.8,
        showfliers: bool = True,
        orient: str = "v",
    ):

        if target not in self.datasets:
            raise KeyError(f"Unknown target '{target}'")

        if variable not in self.continuous + self.discrete:
            raise KeyError(f"Unknown variable '{variable}'")

        dfi = self.datasets[target].copy()

        if feature_labels is None:
            feature_labels = {
                c: self.labels.get(c, c)
                for c in dfi.columns
            }

        groups = sorted(dfi[target].dropna().unique())

        if class_labels is None:
            class_labels = {
                g: str(g)
                for g in groups
            }

        if xlabel is None:
            xlabel = None

        if ylabel is None:
            ylabel = feature_labels.get(variable, variable)

        if title is None:
            title = f"{feature_labels.get(variable, variable)} by {feature_labels.get(target, target)}"

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )

        sns.boxplot(
            x=target,
            y=variable,
            data=dfi,
            palette=palette,
            linewidth=linewidth,
            width=width,
            showfliers=showfliers,
            orient=orient,
            ax=ax,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_xticks(
            range(len(groups)),
            [
                class_labels[g]
                for g in groups
            ],
        )

        if outfile is None:
            outdir = Path("img")
            outdir.mkdir(
                parents=True,
                exist_ok=True,
            )
            outfile = outdir / f"{self.clean[target]}_boxplot_{self.clean[variable]}.png"
        else:
            outfile = Path(outfile)
            outfile.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        fig.tight_layout()
        fig.savefig(outfile)

        plt.close(fig)

        return outfile


    def plot_all_boxplots(
        self,
        variables: list[str] | None = None,
        **kwargs,
    ):

        if variables is None:
            variables = list(self.continuous)

        results = {}

        for target in self.datasets:

            results[target] = {}

            for variable in variables:

                results[target][variable] = self.plot_boxplot(
                    target=target,
                    variable=variable,
                    **kwargs,
                )

        return results
