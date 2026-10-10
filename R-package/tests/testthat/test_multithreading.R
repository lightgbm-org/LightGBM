test_that("getLGBMthreads() and setLGBMthreads() work as expected", {
    # works with integer input
    ret <- setLGBMthreads(2L)
    expect_null(ret)
    expect_equal(getLGBMthreads(), 2L)

    # works with float input
    ret <- setLGBMthreads(1.0)
    expect_null(ret)
    expect_equal(getLGBMthreads(), 1L)

    # setting to any negative number sets max threads to -1
    ret <- setLGBMthreads(-312L)
    expect_null(ret)
    expect_equal(getLGBMthreads(), -1L)
})
