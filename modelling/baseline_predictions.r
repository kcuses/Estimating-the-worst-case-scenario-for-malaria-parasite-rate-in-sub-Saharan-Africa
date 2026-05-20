# ================================================================
# Malaria Baseline Prevalence Mapping Pipeline
# ----------------------------------------------------------------
# This script:
#
# 1. Loads malaria prevalence survey data
# 2. Applies baseline intervention thresholds
# 3. Loads learned contrastive-learning embeddings
# 4. Applies Bayesian posterior coefficients estimated in NumPyro
# 5. Generates continent-wide malaria prevalence predictions
# 6. Produces posterior uncertainty realisations
# 7. Aggregates predictions to administrative units
# 8. Creates raster maps and uncertainty visualisations
#
# ================================================================

# ================================================================
# Required Libraries
# ================================================================

require('raster')
require('rgdal')

library(INLA)
library(RColorBrewer)
library(zoo)
library(scales)
library(matrixStats)
library(doParallel)
library(progress)
library(rnaturalearth)


# ================================================================
# Utility Functions
# ================================================================


# ------------------------------------------------
# Convert longitude/latitude coordinates to XYZ
# coordinates on a sphere
# ------------------------------------------------
ll.to.xyz <- function(ll){

    if(is.null(colnames(ll))){
        colnames(ll) <- c('longitude', 'latitude')
    }

    if(colnames(ll)[1] == 'x' &
       colnames(ll)[2] == 'y'){
        colnames(ll) <- c('longitude', 'latitude')
    }

    if(colnames(ll)[1] == 'lon' &
       colnames(ll)[2] == 'lat'){
        colnames(ll) <- c('longitude', 'latitude')
    }

    ll[, 'longitude'] <- ll[, 'longitude'] * (pi / 180)
    ll[, 'latitude']  <- ll[, 'latitude']  * (pi / 180)

    x = cos(ll[, 'latitude']) *
        cos(ll[, 'longitude'])

    y = cos(ll[, 'latitude']) *
        sin(ll[, 'longitude'])

    z = sin(ll[, 'latitude'])

    return(cbind(x, y, z))
}


# ------------------------------------------------
# Match numerical values to colour palette
# ------------------------------------------------
match.cols <- function(val){

    n = 1000

    col <- data.frame(
        val = seq(min(val),
                  max(val),
                  length.out = n),
        col = colfunc(n)
    )

    out <- rep(NA, length(col))

    for(i in 1:length(val)){

        out[i] <- as.character(
            col[
                which.min(abs(col$val - val[i])),
                'col'
            ]
        )
    }

    return(out)
}


# ------------------------------------------------
# Empirical logit transformation
# ------------------------------------------------
emplogit <- function(Y, N){

    top    = Y * N + 0.5
    bottom = N * (1 - Y) + 0.5

    return(log(top / bottom))
}


# ================================================================
# Colour Palette
# ================================================================

bias = 1

colfunc <- colorRampPalette(c(

    rgb(0/255,0/255,255/255),
    rgb(43/255,65/255,255/255),
    rgb(56/255,109/255,255/255),
    rgb(59/255,157/255,255/255),
    rgb(48/255,207/255,255/255),
    rgb(0/255,255/255,255/255),
    rgb(112/255,255/255,210/255),
    rgb(161/255,255/255,164/255),
    rgb(199/255,255/255,120/255),
    rgb(231/255,255/255,74/255),
    rgb(255/255,255/255,0/255),
    rgb(255/255,213/255,0/255),
    rgb(255/255,166/255,0/255),
    rgb(255/255,123/255,0/255),
    rgb(255/255,77/255,0/255),
    rgb(255/255,0/255,0/255)

), bias = bias)


# ================================================================
# Global Parameters
# ================================================================

nfeat = 1024

threshold_itn  = 0.1
threshold_act  = 0.5
threshold_year = 2008


# ================================================================
# Load and Process Survey Data
# ================================================================

# Load malaria prevalence survey data
d = readRDS('/home/pfpr.rds')

# Remove missing prevalence observations
d = d[!is.na(d$PfPr),]

# Convert prevalence to empirical logit
d$PfPr_logit <- emplogit(
    d$PfPr,
    d$Nexamined
)

# ------------------------------------------------
# Apply baseline intervention thresholds
# ------------------------------------------------
d = d[
    d$itnavg4 < threshold_itn &
    d$act     < threshold_act &
    d$irs     < threshold_itn &
    d$yearqtr < threshold_year,
]


# ================================================================
# Load Learned Contrastive Features
# ================================================================

design_matrix = d[, paste0(
    'feature_',
    0:(nfeat - 1)
)]

colnames(design_matrix) <- paste0(
    'V',
    1:ncol(design_matrix)
)


# ================================================================
# Load Prediction Features
# ================================================================

pred = readRDS('/home/pred.rds')

pred = pred[complete.cases(pred),]

design_matrixp = pred[, paste0(
    'feature_',
    0:(nfeat - 1)
)]

colnames(design_matrixp) <- paste0(
    'V',
    1:ncol(design_matrixp)
)


# ================================================================
# Load Bayesian Posterior Coefficients
# ------------------------------------------------
# These coefficients were estimated using the
# NumPyro Bayesian ridge regression model.
# ================================================================

betas = read.csv(
    '/home/coeffs_0_1.csv',
    header = FALSE
)

# Posterior mean coefficients
betas_mean = as.numeric(colMeans(betas))


# ================================================================
# Predict Baseline Malaria Prevalence
# ================================================================

lp_mean = betas_mean[1] +
    as.matrix(design_matrixp) %*%
    cbind(betas_mean[2:(nfeat + 1)])


# ================================================================
# Visualise Predicted Prevalence
# ================================================================

plot(
    pred$lon,
    pred$lat,
    col = match.cols(
        plogis(as.vector(lp_mean))
    ),
    pch  = 16,
    cex  = 0.1,
    xlab = 'Longitude',
    ylab = 'Latitude'
)


# ================================================================
# Load Administrative and Population Data
# ================================================================

admin = raster('/home/admin2023_0_MG_5K.tif')

population = raster(
    '/home/ihme_corrected_worldpop_All_Ages_3_2023.tif'
)

NAvalue(admin)      <- -9999
NAvalue(population) <- -9999


# ================================================================
# Aggregate Population Raster
# ================================================================

agg_population = aggregate(
    population,
    2,
    sum
)

population = extract(
    agg_population,
    cbind(pred$lon, pred$lat)
)


# ================================================================
# Aggregate Predictions to Administrative Units
# ================================================================

admin_codes = extract(
    admin,
    cbind(pred$lon, pred$lat)
)

unique_admin_codes = unique(admin_codes)

unique_admin_codes =
    unique_admin_codes[
        !is.na(unique_admin_codes)
    ]

lp_admin = unique_admin_codes


for(i in 1:length(unique_admin_codes)){

    pop = population[
        admin_codes == unique_admin_codes[i]
    ]

    pop = pop / sum(pop, na.rm = TRUE)

    mn = sum(
        plogis(
            lp_mean[
                admin_codes ==
                unique_admin_codes[i]
            ]
        ) * pop,
        na.rm = TRUE
    )

    lp_admin[i] <- mn
}


# ================================================================
# Raster Generation Function
# ================================================================

makeraster = function(l){

    r = raster(
        '/home/statelite/master_10km_raster.tif'
    )

    NAvalue(r) = -9999

    r[!is.na(r)] = 0

    cells = cellFromXY(
        r,
        cbind(pred$lon, pred$lat)
    )

    wh = !is.na(cells)

    r[r == 1] = 0

    r[cells] = plogis(l[wh])

    r[r > 2] = NA

    return(r)
}


# ================================================================
# Generate Posterior Realisations
# ------------------------------------------------
# Each posterior sample generates one prevalence
# surface, allowing uncertainty estimation.
# ================================================================

registerDoParallel(20)

lp_sample <- foreach(
    i = 1:nrow(betas),
    .combine = cbind
) %dopar% {

    c1 = as.numeric(betas[i,])

    return(
        c1[1] +
        as.matrix(design_matrixp) %*%
        cbind(c1[2:(nfeat + 1)])
    )
}


# ================================================================
# Convert Posterior Samples into Raster Stack
# ================================================================

pfpr_stack = stack(
    makeraster(lp_sample[,1])
)

for(i in 2:nrow(betas)){

    pfpr_stack[[i]] =
        makeraster(lp_sample[,i])
}


# ================================================================
# Load Incidence Realisations
# ------------------------------------------------
# These raster files were generated separately
# using prevalence-to-incidence conversion.
# ================================================================

incidence_stack = stack(
    paste0(
        '/home/extracted_folder/realisation.',
        1:200,
        '.inc.rate.all.tif'
    )
)


# ================================================================
# Population-Weighted Administrative Aggregation
# ================================================================

pb <- progress_bar$new(
    total = ncol(lp_sample)
)

admin <- raster('/home/admin2023_1_MG_5K.tif')

limits = raster(
    '/home/statelite/pf_limits.tif',
    NAflag = -9999
)

limits[!is.na(limits)] = 1

NAvalue(admin) <- -9999

admin_codes = extract(
    admin,
    cbind(pred$lon, pred$lat)
)

unique_admin_codes = unique(admin_codes)

unique_admin_codes =
    unique_admin_codes[
        !is.na(unique_admin_codes)
    ]

lp_admin <- lp_sample


for(j in 1:ncol(lp_sample)){

    for(i in 1:length(unique_admin_codes)){

        pop = population[
            admin_codes ==
            unique_admin_codes[i]
        ]

        pop = pop / sum(pop, na.rm = TRUE)

        pr = lp_sample[
            admin_codes ==
            unique_admin_codes[i],
            j
        ]

        lp_admin[
            admin_codes ==
            unique_admin_codes[i],
            j
        ] <- sum(pop * pr, na.rm = TRUE)
    }

    pb$tick()
}


# ================================================================
# Compute Posterior Quantiles
# ================================================================

l = rowQuantiles(
    lp_admin,
    probs = c(0.025)
)

m = rowQuantiles(
    lp_admin,
    probs = c(0.5)
)

h = rowQuantiles(
    lp_admin,
    probs = c(0.975)
)


# ================================================================
# Plot Uncertainty Maps
# ================================================================

par(mfrow = c(1,3))

plot(
    pred$lon,
    pred$lat,
    col = match.cols(
        plogis(as.vector(l))
    ),
    pch = 16,
    cex = 0.1
)

plot(
    pred$lon,
    pred$lat,
    col = match.cols(
        plogis(as.vector(m))
    ),
    pch = 16,
    cex = 0.1
)

plot(
    pred$lon,
    pred$lat,
    col = match.cols(
        plogis(as.vector(h))
    ),
    pch = 16,
    cex = 0.1
)


# ================================================================
# Raster Visualisation with Country Borders
# ================================================================

raster_data = makeraster(m)

countries = ne_countries(
    scale = "medium",
    returnclass = "sf"
)

image.plot(
    raster_data,
    legend.width = 1.5,
    legend.mar = 5,
    col = rev(terrain.colors(1000)),
    zlim = c(0, 1)
)

plot(
    countries,
    add = TRUE,
    border = "black",
    col = NA
)


# ================================================================
# Predicted vs Observed Prevalence
# ================================================================

plot(
    lp,
    d$PfPr_logit,
    pch = 16,
    cex = 0.5,
    col = alpha('black', 0.5),
    xlab = 'Predicted',
    ylab = 'Observed'
)

abline(0, 1, col = 'red')
