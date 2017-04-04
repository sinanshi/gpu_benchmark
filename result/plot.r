library(ggplot2)
library(data.table)
allfile <- dir('.')

resfiles <- (allfile[grep(x=allfile, pattern="gemm")])


dts <- list()
for (i in resfiles) {
    txt <- strsplit(i, "[.]")[[1]][1]
    machine <- paste(strsplit(txt, "_")[[1]][1:2], collapse="_")
    datatype <- paste(strsplit(txt, "_")[[1]][3], collapse="_")
    dt <- read.csv(i, header=F, sep=" ")
    dt <- cbind(dt, machine)
    dt <- cbind(dt, datatype)
    dts[[length(dts) + 1]] <- dt
}

dts <- rbindlist(dts)
print(dts)
names(dts) <- c("Size", "Time", "GFlops", "Machine", "DataType")


gg <- ggplot(data.frame(dts), aes(x=Size, y=Time, color=Machine)) + 
    geom_line() + 
    geom_point() + 
    facet_grid(~DataType) + 
    xlab("Matrix Size") + 
    ylab("Time") + 
    labs(title="GPU Performance: CUBLAS Matmul")
print(gg)
