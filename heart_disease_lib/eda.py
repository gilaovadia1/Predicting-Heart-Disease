import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.externals.array_api_extra import nunique


def format_with_percentage(col_x):
    """Format numbers with commas and percentages relative to the column total"""
    total = col_x.iloc[-1]  # Use sum for percentage calculation
    return col_x.apply(lambda x: f"{x:,} ({x / total:.1%})")

class EDA:
    def __init__(self, df, target=None):
        """
        df: pandas DataFrame
        target: name of the target/explained variable (optional)
        """
        self.df = df
        self.target = target

    def NumericDistribution(self, col):
        """
        Prints the distribution of a numeric column.
        If target is provided, prints distribution per target class.
        """
        summary = self.df[col].describe()
        missing_count = self.df[col].isna().sum()
        unique_count = self.df[col].nunique()

        # Convert to a single-row DataFrame
        rotated_summary = summary.to_frame().T

        # Insert missing and unique values
        rotated_summary.insert(1, 'unique values', unique_count)
        rotated_summary.insert(1, 'missing', missing_count)

        out_summary = rotated_summary

        if self.target:
            grouped_summary = self.df.groupby(self.target)[col].describe()

            missing_per_group = self.df.groupby(self.target)[col].apply(lambda x: x.isna().sum())
            unique_per_group = self.df.groupby(self.target)[col].apply(lambda x: x.nunique())

            # Insert missing and unique values
            grouped_summary.insert(1, 'unique values', unique_per_group)
            grouped_summary.insert(1, 'missing', missing_per_group)

            out_summary = pd.concat([rotated_summary, grouped_summary])

        print(out_summary)

    def PlotDistributions(self, col):
        """
        Plot 3 distributions of a numeric column:
        1. Overall
        2. Where target = 0/Absence
        3. Where target = 1/Presence
        """
        if self.target:
            ncols=3
        else:
            ncols=1

        fig, axes = plt.subplots(1, ncols, figsize=(18, 5))
        overall_data = self.df[col]

        # Overall distribution
        sns.histplot(overall_data, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'{col} - Overall')

        if self.target:
            target_levels = self.df[self.target].unique()
            target0_data = self.df[self.df[self.target] == target_levels[0]][col]
            target1_data = self.df[self.df[self.target] == target_levels[1]][col]

            # Distribution for target = 0
            sns.histplot(target0_data, kde=True, ax=axes[1], color='green')
            axes[1].set_title(f'{col} - {self.target}: {target_levels[0]}')

            # Distribution for target = 1
            sns.histplot(target1_data, kde=True, ax=axes[2], color='red')
            axes[2].set_title(f'{col} - {self.target}: {target_levels[1]}')

        plt.tight_layout()
        plt.show()

    def ObjectDistribution(self, col):
        # Overall counts
        overall = self.df[col].value_counts().rename('Count')

        # Counts by target
        target_counts = self.df.groupby([col, self.target]).size().unstack(fill_value=0)

        # Combine into summary
        summary = pd.concat([overall, target_counts], axis=1)

        # Add Total row
        total_row = summary.sum()
        total_row.name = 'Total'
        summary = pd.concat([summary, total_row.to_frame().T])

        # Format numbers with commas and percentages
        summary_formatted = summary.apply(format_with_percentage)

        print(summary_formatted)

    # def ObjectDistribution(self, col):
    #     overall = self.df[col].value_counts().rename('Count')
    #     target_counts = self.df.groupby([col, self.target]).size().unstack(fill_value=0)
    #     summary = pd.concat([overall, target_counts], axis=1)
    #     summary_formatted = summary.apply(self.format_with_percentage)
    #
    #     print(summary_formatted)

    def VariableSummary(self):
        for col in self.df.columns:
            print(f'========== {col} ==========')
            if pd.api.types.is_object_dtype(self.df[col]) or self.df[col].nunique() <= 5:
                if pd.api.types.is_object_dtype(self.df[col]):
                    print('Type: Object')
                else:
                    print('Type: Numeric with less than 5 unique values - analyzing as object')
                    self.df[col] = self.df[col].astype(object)
                self.ObjectDistribution(col)
            elif is_numeric_dtype(self.df[col]):
                print("Type: Numeric")
                self.NumericDistribution(col)
                self.PlotDistributions(col)
            else:
                print('!!! WARNING: COLUMN IS NOT NUMERIC OR OBJECT!!! (no summary printed)')