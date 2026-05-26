# ============================================================
# STANDALONE PREVALENCE -> INCIDENCE CONVERSION
#
# Public reproducible implementation
#
# This script:
#
# 1. Learns an empirical prevalence-to-incidence mapping
#    using calibration rasters
#
# 2. Demonstrates a small synthetic example workflow
#
# 3. Converts prevalence rasters to incidence rasters
#
# ============================================================

library(terra)
library(mgcv)

# ============================================================
# DIRECTORIES
# ============================================================

# Directory containing prevalence realisations
prevalence_dir <- "prevalence_rasters"

# Calibration incidence rasters
#
# Used ONLY to empirically learn the
# prevalence -> incidence relationship
#
calibration_incidence_dir <- "incidence_rasters"

# Output directory
output_dir <- "incidence_rasters_calibrated"

dir.create(
  output_dir,
  showWarnings = FALSE
)

# ============================================================
# SETTINGS
# ============================================================

n_realisations <- 200

# Realisations used to learn mapping
calibration_ids <- c(
  1,
  25,
  50,
  100,
  150,
  200
)

# ============================================================
# STEP 1:
# BUILD CALIBRATION DATASET
# ============================================================

cat("====================================\n")
cat("BUILDING CALIBRATION DATASET\n")
cat("====================================\n")

all_prev <- c()
all_inc  <- c()

for(i in calibration_ids){
  
  cat("Loading calibration realisation:", i, "\n")
  
  # ----------------------------------------------------------
  # LOAD PREVALENCE RASTER
  # ----------------------------------------------------------
  
  prev_file <- paste0(
    prevalence_dir,
    "/realisation_",
    i,
    ".tif"
  )
  
  prev_rast <- rast(prev_file)
  
  # ----------------------------------------------------------
  # LOAD CALIBRATION INCIDENCE RASTER
  # ----------------------------------------------------------
  
  inc_file <- paste0(
    calibration_incidence_dir,
    "/realisation.",
    i,
    ".inc.rate.all.tif"
  )
  
  inc_rast <- rast(inc_file)
  
  # ----------------------------------------------------------
  # EXTRACT VALUES
  # ----------------------------------------------------------
  
  prev_vals <- values(prev_rast)
  inc_vals  <- values(inc_rast)
  
  wh <- !is.na(prev_vals) &
    !is.na(inc_vals)
  
  prev_vals <- prev_vals[wh]
  inc_vals  <- inc_vals[wh]
  
  # ----------------------------------------------------------
  # RANDOM SUBSAMPLING
  #
  # Prevents excessive memory usage
  # ----------------------------------------------------------
  
  n_sample <- min(
    100000,
    length(prev_vals)
  )
  
  idx <- sample(
    seq_along(prev_vals),
    n_sample
  )
  
  all_prev <- c(
    all_prev,
    prev_vals[idx]
  )
  
  all_inc <- c(
    all_inc,
    inc_vals[idx]
  )
}

# ============================================================
# CREATE CALIBRATION DATAFRAME
# ============================================================

calibration_df <- data.frame(
  prevalence = all_prev,
  incidence  = all_inc
)

cat(
  "Calibration samples:",
  nrow(calibration_df),
  "\n"
)

# ============================================================
# STEP 2:
# FIT PREVALENCE -> INCIDENCE MODEL
# ============================================================

cat("\n====================================\n")
cat("FITTING CALIBRATION MODEL\n")
cat("====================================\n")

gam_fit <- gam(
  incidence ~ s(prevalence, k = 10),
  data = calibration_df
)

summary(gam_fit)

# ============================================================
# SAVE MODEL
# ============================================================

saveRDS(
  gam_fit,
  paste0(
    output_dir,
    "/prevalence_to_incidence_gam.rds"
  )
)

# ============================================================
# DEFINE CONVERSION FUNCTION
# ============================================================

pfpr_to_incidence <- function(pfpr_values){
  
  predicted <- predict(
    gam_fit,
    newdata = data.frame(
      prevalence = pfpr_values
    )
  )
  
  # ----------------------------------------------------------
  # CLAMP VALUES
  # ----------------------------------------------------------
  
  predicted[predicted < 0] <- 0
  
  predicted[predicted > 1] <- 1
  
  return(predicted)
}

# ============================================================
# SYNTHETIC EXAMPLE WORKFLOW
# ============================================================

cat("\n====================================\n")
cat("SYNTHETIC EXAMPLE WORKFLOW\n")
cat("====================================\n")

# Example prevalence values
test_pfpr <- c(
  0.01,
  0.10,
  0.25,
  0.50,
  0.75
)

# Convert prevalence -> incidence
test_incidence <- pfpr_to_incidence(
  test_pfpr
)

# Create results table
test_results <- data.frame(
  PfPR = test_pfpr,
  Incidence = test_incidence
)

# Print results
print(test_results)

# Save results
write.csv(
  test_results,
  paste0(
    output_dir,
    "/synthetic_example_conversion.csv"
  ),
  row.names = FALSE
)

# ============================================================
# OPTIONAL:
# VISUALISE FITTED FUNCTION
# ============================================================

png(
  paste0(
    output_dir,
    "/calibration_curve.png"
  ),
  width = 1200,
  height = 800
)

plot(
  gam_fit,
  shade = TRUE,
  main = "Prevalence-to-Incidence Calibration"
)

dev.off()

# ============================================================
# STEP 3:
# GENERATE INCIDENCE RASTERS
# ============================================================

cat("\n====================================\n")
cat("GENERATING INCIDENCE RASTERS\n")
cat("====================================\n")

for(i in 1:n_realisations){
  
  cat("Processing realisation:", i, "\n")
  
  # ----------------------------------------------------------
  # LOAD PREVALENCE RASTER
  # ----------------------------------------------------------
  
  prev_file <- paste0(
    prevalence_dir,
    "/realisation_",
    i,
    ".tif"
  )
  
  prev_rast <- rast(prev_file)
  
  prev_vals <- values(prev_rast)
  
  # ----------------------------------------------------------
  # APPLY CONVERSION FUNCTION
  # ----------------------------------------------------------
  
  predicted_incidence <- rep(
    NA,
    length(prev_vals)
  )
  
  wh <- !is.na(prev_vals)
  
  predicted_incidence[wh] <- pfpr_to_incidence(
    prev_vals[wh]
  )
  
  # ----------------------------------------------------------
  # CREATE OUTPUT RASTER
  # ----------------------------------------------------------
  
  inc_rast <- prev_rast
  
  values(inc_rast) <- predicted_incidence
  
  # ----------------------------------------------------------
  # SAVE OUTPUT
  # ----------------------------------------------------------
  
  output_file <- paste0(
    output_dir,
    "/realisation.",
    i,
    ".inc.rate.all.tif"
  )
  
  writeRaster(
    inc_rast,
    output_file,
    overwrite = TRUE
  )
}

cat("\nDone.\n")