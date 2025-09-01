import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 轉換成中文字體
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]


def visualization_page(df, id_column, date_column, category_column, numeric_columns, visualization_type):
    # Exclude id columns from the list of columns
    columns = [i for i in df.columns if i not in id_column]

    plot_base64 = None
    
    if visualization_type == "Line Chart" and numeric_columns:
        if len(numeric_columns) >= 1:
            plt.figure(figsize=(10, 6), dpi=150)
            sns.lineplot(data=df[columns])
            plt.title("Line Chart")

            # Save plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Encode the plot to base64
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
    elif visualization_type == "Bar Chart" and category_column:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=category_column, data=df)
        plt.title("Bar Chart")
        plt.xticks(rotation=80, fontsize=15)

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode the plot to base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    elif visualization_type == "Scatter Plot" and numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=numeric_columns[0], y=numeric_columns[1], data=df)
        plt.title("Scatter Plot")

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode the plot to base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    elif visualization_type == "Time Series" and len(date_column + numeric_columns) >= 2:
        if date_column and numeric_columns[0]:
            plt.figure(figsize=(10, 6)) 
            sns.lineplot(x=date_column, y=numeric_columns[0], data=df)
            plt.title(f"Time Series of {numeric_columns[0]} over {date_column}")
            plt.xticks(rotation=45)

            # Save plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Encode the plot to base64
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

    return plot_base64