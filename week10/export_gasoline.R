# Install the pls package if it is not already installed
if (!requireNamespace("pls", quietly = TRUE)) {
  install.packages("pls", repos = "https://cloud.r-project.org")
}

# Load the package
library(pls)

# Load the gasoline dataset
data(gasoline, package = "pls")

# Convert to a data frame and export as CSV
gasoline_df <- as.data.frame(gasoline)
write.csv(gasoline_df, "gasoline.csv", row.names = FALSE)

cat("Successfully exported gasoline.csv to:", getwd(), "\\n")
