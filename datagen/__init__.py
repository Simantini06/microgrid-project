"""Stage 1 - synthetic microgrid dataset generation.

Produces a clearly-labelled, physically-motivated synthetic hourly time series
(weather -> solar & wind generation, demand, and time-of-use grid price) plus a
data dictionary. Every value is reproducible from a fixed random seed. See
`datagen.generate` for the physical assumptions behind each column.
"""
