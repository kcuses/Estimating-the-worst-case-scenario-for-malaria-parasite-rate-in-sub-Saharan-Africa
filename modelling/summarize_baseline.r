
# Code for summarizing the 10k realizations made by KC

# Add required libraries
library(terra)

# Define the windows flag used in the aggregation function (setting to zero implies we'd use linux here
windows.flag <- 1

if (windows.flag == 1) { root <- 'Z:/' } else { root <- '/mnt/Z/' }

# Define runtime parameters and checks
n.realizations.per.year <- 200 # each year will be checked to see if it has a full set
UCI.threshold <- 0.975
LCI.threshold <- 0.025

# Define the input and output paths
input.path     <- paste0(root, 'temp/inc_all_age/')
output.path    <-  paste0(root, 'temp/summarized/')
limits.folder  <- '20241002'#'20231204'         # this will hold the Pf limits folders (step 10)

# Define the year to use as the baseline (will link to pop and limits)
year <- 2024

# Add the secondary label to the filenames now that the output folders have been established (i.e., we don't put 'infants' into the output raster names)
metric          <- 'incidence_rate'
secondary.label <- age.label <- 'all_age'
geog.label      <- 'Africa_admin1'
  
# Define output prefixes for tables and rasters 
output.mean.prefix   <- paste(metric, '_rmean_', geog.label, '_', sep='')
output.median.prefix <- paste(metric, '_median_', geog.label, '_', sep='')
output.LCI.prefix    <- paste(metric, '_LCI_', geog.label, '_', sep='')
output.UCI.prefix    <- paste(metric, '_UCI_', geog.label, '_', sep='')
output.stdev.prefix  <- paste(metric, '_stdev_', geog.label, '_', sep='')

output.mean.count.prefix   <- paste(metric, '_rmean_count_', geog.label, '_', sep='')
output.median.count.prefix <- paste(metric, '_median_count_', geog.label, '_', sep='')
output.LCI.count.prefix    <- paste(metric, '_LCI_count_', geog.label, '_', sep='')
output.UCI.count.prefix    <- paste(metric, '_UCI_count_', geog.label, '_', sep='')
output.stdev.count.prefix  <- paste(metric, '_stdev_count_', geog.label, '_', sep='')

output.table.prefix  <- paste(metric, '_table_', geog.label, '_', sep='')
untreated.output.table.prefix  <- paste('untreated_', metric, '_table_', geog.label, '_', sep='')

admin.filename                 <- paste(root, 'master_geometries/Admin_Units/Global/MAP/2023/MG_5K/Rasters/admin2023_1_MG_5K.tif', sep='')
config.filename                <- paste(root, 'master_geometries/Admin_Units/Global/MAP/2023/MG_5K/Rasters/admin2023_1_MG_5K_config.csv', sep='') # this is needed for adding metadata columns to the output table
config.name.column             <- 'Name_1'
config.ID.column               <- 'ID_1'   # For admin 0 or 1 or 2
config.iso3.column             <- 'ISO'
write.raster.flag              <- 0

# Because only one (usually year-specific) file is needed for pop and limits we can pre-bake them in using just the year
population.path                <-  paste(root, 'GBD2024/Processing/Stages/05_Raster_Populations/Checkpoint_Outputs/Output_Pop_Unmasked_5k/', sep='')
population.prefix              <- 'ihme_corrected_worldpop_'
population.suffix              <- '.tif'  # the filename characters and extension information that follows the year
population.path.and.prefix     <- paste(population.path, population.prefix, 'All_Ages_3_', sep='')

# Set the species-specific input paths
limits.path.and.prefix         <- paste(root, 'GBD2024/Processing/Stages/10_Create_PostHoc_Masks/Checkpoint_Outputs/', limits.folder, '/Pf/Pf_PostHoc_ZeroAPIForcing_Limits_', sep='')
limits.suffix                  <- '_5k.tif'

# Assemble the input filenames
population.filename <- paste(population.path.and.prefix, year, population.suffix, sep='')
limits.filename     <- paste(limits.path.and.prefix, year, limits.suffix, sep='')


##### Process the  realizations
realizations.path <- input.path 

# Clean up the output folders to remove any of the additional files added by GIS programs (e.g., .tif.ovr and .tif.aux.xml) - these will mess up the file searches later
files.to.delete.vec <- list.files(realizations.path, pattern = '.ovr', full.names = TRUE, recursive=TRUE)
file.remove(files.to.delete.vec)
files.to.delete.vec <- list.files(realizations.path, pattern = '.xml', full.names = TRUE, recursive=TRUE)
file.remove(files.to.delete.vec)

# Create the list of realizations to process for the user-defined year  
realizations.list      <- list.files(realizations.path, pattern = '.tif', recursive=T)
n.realizations.in.list <- length(realizations.list)

mean.filename       <- paste(output.path, output.mean.prefix, 'baseline.tif', sep='')
median.filename     <- paste(output.path, output.median.prefix,'baseline.tif', sep='')
LCI.filename        <- paste(output.path, output.LCI.prefix, 'baseline.tif', sep='')
UCI.filename        <- paste(output.path, output.UCI.prefix, 'baseline.tif', sep='')
stdev.filename      <- paste(output.path, output.stdev.prefix, 'baseline.tif', sep='')

mean.count.filename       <- paste(output.path, output.mean.count.prefix, 'baseline.tif', sep='')
median.count.filename     <- paste(output.path, output.median.count.prefix, 'baseline.tif', sep='')
LCI.count.filename        <- paste(output.path, output.LCI.count.prefix, 'baseline.tif', sep='')
UCI.count.filename        <- paste(output.path, output.UCI.count.prefix, 'baseline.tif', sep='')
stdev.count.filename      <- paste(output.path, output.stdev.count.prefix, 'baseline.tif', sep='')

table.filename      <- paste(output.path, output.table.prefix, 'baseline.csv', sep='')
table.full.filename <- paste(output.path, output.table.prefix, 'full_baseline.csv', sep='')

# Read in the config file
config.table <- read.csv(config.filename)
cols.to.keep <- c(config.name.column, config.iso3.column, config.ID.column) 
config.table <- config.table[cols.to.keep]
config.table <- config.table[!is.na(config.table[,3]),] # adjustment for cases where the config table has extra rows

#rename the config columns for use later in the code
colnames(config.table)[1] <- 'Name'
colnames(config.table)[2] <- 'ISO3'
colnames(config.table)[3] <- 'ID'

# Establish a template file for subsetting the global rasters (by choosing a realization the whole stack won't need to be subset)
template.filename <- paste(realizations.path, realizations.list[1], sep='')
template <- rast(template.filename)
e        <- ext(template)

# Read and crop in the global rasters
admin <- rast(admin.filename)
admin <- crop(admin, e, snap="near", extend=FALSE)	
pop   <- rast(population.filename)
pop   <- crop(pop, e, snap="near", extend=FALSE)	
limit <- rast(limits.filename)
limit <- crop(limit, e, snap="near", extend=FALSE)	

## Note that at this point the template and the pop, admin, and limit files are not matched as the template is at 10 km resolution

# Catch to remove any negative population values in case -9999 was introduced in the production chain
pos.pop.vals <- pop > 0
pop          <- pop * pos.pop.vals
	
# Set NA values to zero (because it can't hurt)
pop[is.na(pop)] <- 0
limit[is.na(limit)] <- 0

####### Create the list of countries to process and establish the first output columns  #######

# Use the admin raster to define the set of countries
country.vec <- unique(admin)
n.countries <- nrow(country.vec)

# Predetermine how many rows and columns the output table will contain
n.output.rows <- n.countries
n.output.cols <- 4 + length(cols.to.keep) + n.realizations.per.year # 3 for the standard number of output columns

# Define the results table - this will contain columns for each realization, but those will be collapsed prior to writing out to only retain the mean, median, LCI, and UCI values
results.table <- as.data.frame(matrix(NA, nrow = n.output.rows, ncol = n.output.cols))
    
# Have to add an additional header column if a config file is used (and thus names are attached to the output rows)
colnames(results.table)[1] <- 'Name'
colnames(results.table)[2] <- 'ISO3'
colnames(results.table)[3] <- 'ID'
colnames(results.table)[4] <- 'Year'
colnames(results.table)[5] <- 'Age'
colnames(results.table)[6] <- 'Pop'
colnames(results.table)[7] <- 'PAR'
n.header.cols <- 7
 
# Make an initial pass through the endemic countries to structure the initial (header) columns of the table
for (a in 1:n.countries){

  ID.val <- country.vec[a,] # Key change related to Terra here, add the comma after the A because Terra's unique returns a 2D table

  id.in.table <- ID.val %in% config.table$ID

  if (id.in.table == TRUE){
    config.row <- config.table[config.table$ID == ID.val,]
    name.val   <- as.character(config.row$Name)
    iso3.val   <- as.character(config.row$ISO3)
    results.table$Name[a] <- name.val
    results.table$ISO3[a] <- iso3.val 
  } else {
    results.table$Name[a] <- 'NA'
    results.table$ISO3[a] <- 'NA'
  }
  results.table$Year[a] <- toString(year)
  results.table$Age[a]  <- age.label 
  results.table$ID[a]   <- ID.val
      
} # end A loop

# Pre-bake the population at risk (PAR) raster 
PAR <- pop * limit # raster * raster

# Derive the population and PAR values
pop.zonal <- zonal(pop, admin, fun='sum')
PAR.zonal <- zonal(PAR, admin, fun='sum')

# Place the resulting zonal sums within the results table
results.table[,6] <- pop.zonal[,2]
results.table[,7] <- PAR.zonal[,2]

####### Process the realizations  #######
for (d in 1:n.realizations.in.list){
#for (d in 1:5){
  #d <- 1
      
  # Fetch one filename to open and load it.
  realization.filename <- paste(realizations.path, realizations.list[d], sep='') 

  realization <- rast(realization.filename)
  
  # do the resample
  realization <- resample(realization, pop, method="near")

  if (d == 1) { 
    if (write.raster.flag == 1) {
      # For the eventual mean calculation
      sum.of.realizations <- realization # the sum, stdev, and CI rasters (if made) will all stay in units of rate#

      # Parts for the moving StDev calculation
      oldM <- realization
      newM <- realization
      oldS <- realization * 0.0

      # For create the realization stack 
      realization.stack <- realization
    }
  } else { 
        
    if (write.raster.flag == 1) {
      # For the eventual mean calculation
      sum.of.realizations <- sum.of.realizations + realization

      # Parts for the moving StDev calculation
      newM <- oldM + ((realization - oldM) / d)
      newS <- oldS + ((realization - oldM) * (realization - newM))
      oldM <- newM # set up for next iteration
      oldS <- newS # set up for next iteration

      # Add this realization to the raster stack
      realization.stack <- c(realization.stack, realization)
    }
  }

  # Convert the rate raster into a count raster - the table will initially be filled with counts and only converted back to rates at the end
  realization           <- realization * pop

  # Calculate the zonal stats for this realization
  realization.zonal           <- zonal(realization, admin, fun='sum', na.rm=TRUE)

  # Place the resulting zonal sums within the results table
  col.pos <- n.header.cols + d
  results.table[,col.pos] <- realization.zonal[,2]

} # end D loop


# Establish the output table (to be written) by pre-populating the header columns 
output.table <- results.table[,1:(n.header.cols)]

# Make a summary table (mean, lci, median, uci)
data.only <- results.table[,(n.header.cols + 1):n.output.cols]
cis <- apply(data.only, 1, quantile, probs=c(0.025, 0.5, 0.975), na.rm=TRUE)
output.table[,(n.header.cols + 1)] <- rowMeans(data.only, na.rm=TRUE)
output.table[,(n.header.cols + 2):(n.header.cols + 4)] <- t(cis)

colnames(output.table)[(n.header.cols + 1)] <- paste(metric, '_rmean', sep='')
colnames(output.table)[(n.header.cols + 2)] <- paste(metric, '_LCI', sep='')
colnames(output.table)[(n.header.cols + 3)] <- paste(metric, '_median', sep='')
colnames(output.table)[(n.header.cols + 4)] <- paste(metric, '_UCI', sep='')

# Double check that we have the expected number of rows - for testing
#n.rows <- dim(output.table)[1]
#row.count.test <- (n.output.rows == n.rows)

# Change the column names for the full table
colnames(results.table) <- append(colnames(results.table)[1:n.header.cols], as.character(rep(1:n.realizations.per.year)))

# Convert the count values to rates
output.table[,(n.header.cols + 1):(n.header.cols + 4)] <- output.table[,(n.header.cols + 1):(n.header.cols + 4)] / pop.zonal[,2]
results.table[,(n.header.cols + 1):n.output.cols] <- results.table[,(n.header.cols + 1):n.output.cols] / pop.zonal[,2]

# For reasons I don't understand I get an error when I write out the .csv (there's something weird with the numbers, somehow)
#output.table$Name <- as.character(output.table$Name)
#output.table$ID <- as.numeric(output.table$ID)

# Append the Rmean the all_realizations table
n.cols      <- dim(results.table)[2]
header.cols <- results.table[,1:(n.header.cols)]
rmean.col   <- output.table[,(n.header.cols + 1)]
data.cols   <- results.table[,(n.header.cols + 1):n.cols]
results.table <- cbind(header.cols, rmean.col)
results.table <- cbind(results.table, data.cols)
colnames(results.table)[(n.header.cols + 1)] <- 'rmean'

# Apply the omit list
#omit.list           <- c(1988, 1993, 2004, 2035, 2042, 2046, 2047, 2061, 2067, 2085, 2086, 2088, 2089, 2092, 2101, 2103 ,2105, 2109, 2114, 2123, 2135, 2148, 2160, 2162, 2164, 2169, 2174, 2185, 2191, 2192, 2203, 2204, 2222, 2225) # for Africa only - hack addition
#endemic.vec         <- country.vec[!country.vec %in% omit.list]
#results.table <- results.table[results.table$ID %in% endemic.vec,]
#output.table <- output.table[output.table$ID %in% endemic.vec,]

# Write the summary and full realizations tables
write.csv(output.table, table.filename, row.names=F)
write.csv(results.table, table.full.filename, row.names=F)


if (write.raster.flag == 1) {
  # Calculate mean and StDev rasters
  mean.raster  <- sum.of.realizations / n.realizations.in.list
  stdev.raster <- sqrt(newS / (n.realizations.in.list - 1))

  writeRaster(mean.raster, mean.filename, overwrite=TRUE) # Write the mean raster
  writeRaster(stdev.raster, stdev.filename, overwrite=TRUE) # Write the stdev raster

  # Calculate CI rasters
  CI.rasters <- app(realization.stack, fun=function(x) {quantile(x, probs = c(LCI.threshold, 0.5, UCI.threshold), na.rm=TRUE)}) # SLOW step, 394.17 sec (6.5 min) elapsed on desktop
  LCI.raster    <- subset(CI.rasters, 1)
  median.raster <- subset(CI.rasters, 2)
  UCI.raster    <- subset(CI.rasters, 3)
  writeRaster(LCI.raster, LCI.filename, overwrite=TRUE) # Write the LCI raster
  writeRaster(median.raster, median.filename, overwrite=TRUE) # Write the median raster
  writeRaster(UCI.raster, UCI.filename, overwrite=TRUE) # Write the UCI raster  

  # Make and write all count rasters 
  mean.raster   <- mean.raster * pop
  stdev.raster  <- stdev.raster * pop
  LCI.raster    <- LCI.raster * pop
  median.raster <- median.raster * pop
  UCI.raster    <- UCI.raster * pop

  writeRaster(mean.raster, mean.count.filename, overwrite=TRUE) # Write the mean raster
  writeRaster(stdev.raster, stdev.count.filename, overwrite=TRUE) # Write the stdev raster
  writeRaster(LCI.raster, LCI.count.filename, overwrite=TRUE) # Write the LCI raster
  writeRaster(median.raster, median.count.filename, overwrite=TRUE) # Write the median raster
  writeRaster(UCI.raster, UCI.count.filename, overwrite=TRUE) # Write the UCI raster  
} # end write raster flag
