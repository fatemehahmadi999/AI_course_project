import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.common.GeneralTools

// =====================
// SETTINGS
// =====================

def tileSize = 512
def downsample = 1.0

def outputDir = buildFilePath(PROJECT_BASE_DIR, "data", "exported_dataset")
def imageDir = buildFilePath(outputDir, "images")
def maskDir  = buildFilePath(outputDir, "masks")

mkdirs(imageDir)
mkdirs(maskDir)

// =====================
// CURRENT IMAGE
// =====================

def imageData = getCurrentImageData()
def imageName = GeneralTools.stripExtension(getProjectEntry().getImageName())

println "Exporting image: " + imageName

// =====================
// LABEL MASK SERVER
// =====================
// Label encoding:
// 0 = background
// 1 = tumor
// 2 = blood vessel
// 3 = airway

def labelServer = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(0, ColorTools.WHITE)
        .addLabel("tumor", 1)
        .addLabel("blood vessel", 2)
        .addLabel("airway", 3)
        .downsample(downsample)
        .multichannelOutput(false)
        .build()

// =====================
// EXPORT TILES AND MASKS
// =====================

new TileExporter(imageData)
        .downsample(downsample)
        .imageExtension(".png")
        .tileSize(tileSize)
        .annotatedTilesOnly(true)
        .overlap(0)
        .labeledServer(labelServer)
        .writeTiles(outputDir)

println "Export finished."
println "Output saved to: " + outputDir