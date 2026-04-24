#' @name lgb.plot.tree
#' @title Plot LightGBM trees.
#' @description The \code{lgb.plot.tree} function creates a DiagrammeR plot of one or more LightGBM trees.
#' @param model a \code{lgb.Booster} object.
#' @param tree An integer vector of tree indices that should be visualized IMPORTANT:
#' the tree index in lightgbm is zero-based, i.e. use tree = 0 for the first tree in a model.
#' @param rules a list of rules to replace the split values with feature levels.
#' @param render a logical flag for whether the graph should be rendered (see Value).
#' @param plot_width the width of the diagram in pixels.
#' @param plot_height the height of the diagram in pixels.
#'
#' @return
#' When \code{render = TRUE}:
#' returns a rendered graph object which is an \code{htmlwidget} of class \code{grViz}.
#' Similar to ggplot objects, it needs to be printed to see it when not running from command line.
#'
#' When \code{render = FALSE}:
#' silently returns a graph object which is of DiagrammeR's class \code{dgr_graph}.
#' This could be useful if one wants to modify some of the graph attributes
#' before rendering the graph with \code{\link[DiagrammeR]{render_graph}}.
#'
#' @details
#' The \code{lgb.plot.tree} function creates a DiagrammeR plot of a single LightGBM tree.
#' The tree is extracted from the model and displayed as a directed graph.
#' The nodes are labelled with the feature, split value, gain, count and value.
#' The edges are labelled with the decision type and split value.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' # Example One
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 0.3
#'   , num_leaves = 5L
#' )
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#' )
#'
#' # Plot the first tree
#' lgb.plot.tree(model, 0L)
#'
#' # Plot the first and fifth trees
#' lgb.plot.tree(model, c(0L,4L))
#'
#' # Example Two - model uses categorical features
#' data(bank, package = "lightgbm")
#'
#' # We are dividing the dataset into two: one train, one validation
#' bank_train <- bank[1L:4000L, ]
#' bank_test <- bank[4001L:4521L, ]
#'
#' # We must now transform the data to fit in LightGBM
#' # For this task, we use lgb.convert_with_rules
#' # The function transforms the data into a fittable data
#' bank_rules <- lgb.convert_with_rules(data = bank_train)
#' bank_train <- bank_rules$data
#'
#' # Remove 1 to label because it must be between 0 and 1
#' bank_train$y <- bank_train$y - 1L
#'
#' # Data input to LightGBM must be a matrix, without the label
#' my_data_train <- as.matrix(bank_train[, 1L:16L, with = FALSE])
#'
#' # Creating the LightGBM dataset with categorical features
#' # The categorical features can be passed to lgb.train to not copy and paste a lot
#' dtrain <- lgb.Dataset(
#'   data = my_data_train
#'   , label = bank_train$y
#'   , categorical_feature = c(2L, 3L, 4L, 5L, 7L, 8L, 9L, 11L, 16L)
#' )
#'
#' # Train the model with 5 training rounds
#' params <- list(
#'   objective = "binary"
#'   , metric = "l2"
#'   , learning_rate = 0.1
#'   , num_leaves = 5L
#' )
#' model_bank <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#' )
#'
#' # Plot the first two trees in the model without specifying "rules"
#' lgb.plot.tree(model_bank, tree = 0L:1L)
#'
#' # Plot the first two trees in the model specifying "rules"
#' lgb.plot.tree(model_bank, rules = bank_rules$rules, tree = 0L:1L)
#'
#' }
#' @importFrom data.table := fcoalesce fifelse setnames
#' @importFrom DiagrammeR add_global_graph_attrs create_edge_df create_graph create_node_df render_graph
#' @export
lgb.plot.tree <- function(model
                          , tree
                          , rules = NULL
                          , render = TRUE
                          , plot_width = NULL
                          , plot_height = NULL
                          ) {
    # check model is lgb.Booster
    if (!.is_Booster(x = model)) {
        stop("lgb.plot.tree: model should be an ", sQuote("lgb.Booster"))
    }
    # check DiagrammeR is available
    if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
        stop("lgb.plot.tree: DiagrammeR package is required",
            call. = FALSE
        )
    }
    # all elements of tree must be integers
    if (!inherits(tree, c("integer", "numeric")) || !all(tree %% 1L == 0L)) {
      stop(sprintf("lgb.plot.tree: 'tree' must only contain integers."))
    }
    # extract data.table model structure
    modelDT <- lgb.model.dt.tree(model)
    # check that all values of tree are greater than zero and less than or equal to the maximum tree index in the model
    if (!all(tree >= 0L & tree <= max(modelDT$tree_index))) {
      stop(
        "lgb.plot.tree: All values of 'tree' should be between 0 and the total number of trees in the model minus one ("
        , max(modelDT$tree_index)
        , ")."
        )
    }
    # filter modelDT to just the rows for the selected trees
    modelDT <- modelDT[tree_index %in% tree]
    # change some column names to shorter and more diagram friendly versions
    data.table::setnames(
      modelDT
      , old = c("tree_index", "split_feature", "threshold", "split_gain")
      , new = c("Tree", "Feature", "Split", "Gain")
    )
    # the output from "lgb.model.dt.tree" follows these rules
    # "leaf_value" and "leaf_count" are only populated for leaves (NA for internal splits)
    # "internal_value" and "internal_count" are only populated for splits (NA for leaves)
    # for the diagram, combine leaf_value and internal_value into a single column called "Value"
    modelDT[, Value := data.table::fcoalesce(leaf_value, internal_value)]
    # for the diagram, combine leaf_count and internal_count into a single column called "Count"
    modelDT[, Count := data.table::fcoalesce(leaf_count, internal_count)]
    # "Feature" is only present for splits, it is NA for leaves
    # Use the text "Leaf" to denote leaves in the diagram
    modelDT[is.na(Feature), Feature := "Leaf"]
    # within each tree, "Node" holds a unique index for each split and leaf
    # for splits, Node = split_index (already populated by lgb.model.dt.tree as an integer)
    # for leaves, Node = max(split_index) for that tree, plus the leaf_index plus one
    # plus one is needed as leaf_index starts at zero within each tree
    modelDT[, Node := split_index]
    modelDT[, Node := data.table::fifelse(!is.na(Node), Node, max(Node, na.rm = TRUE) + leaf_index + 1L), by = Tree]
    # create an ID column to uniquely identify each Node in the diagram (even if there are multiple trees)
    # concatenate Tree and Node, e.g. "0-3" is the third node in the zeroth tree
    modelDT[, ID := paste(Tree, Node, sep = "-")]
    modelDT[, parent := node_parent][is.na(parent), parent := leaf_parent]
    # each split node is parent to two "descendent" nodes
    # column "Yes" will hold the ID of the first descendent node
    # column "No" will hold the ID of the second descendent node
    modelDT[, Yes := ID[match(Node, parent)], by = Tree]
    # reverse the order of modelDT
    # so the match now finds the second descendent node
    modelDT <- modelDT[rev(seq_len(.N))]
    modelDT[, No := ID[match(Node, parent)], by = Tree]
    # which way do the NA's go (this path will get a thicker arrow)
    modelDT[default_left == "TRUE", Missing := Yes]
    modelDT[default_left == "FALSE", Missing := No]
    # create the label text for each node
    # for leaves include the Gain, rounded to 6 s.f. for display
    # round the Value to 6 s.f. for display
    modelDT[, label := paste0(
        Feature
        , "\nCount: "
        , Count
        , data.table::fifelse(Feature == "Leaf", "", "\nGain: ")
        , data.table::fifelse(Feature == "Leaf", "", as.character(round(Gain, 6L)))
        , "\nValue: "
        , round(Value, 6L)
    )]
    # ensure the initial split in each tree is correctly labelled
    modelDT[Node == 0L, label := paste0("Tree ", Tree, "\n", label)]
    # style nodes with rectangles for splits and ovals for leaves
    modelDT[, shape := "rectangle"][Feature == "Leaf", shape := "oval"]
    # style Nodes with the same colours as xgboost's xgb.plot.trees
    modelDT[, filledcolor := "Beige"][Feature == "Leaf", filledcolor := "Khaki"]
    # create the diagram nodes
    nodes <- DiagrammeR::create_node_df(
        n         = nrow(modelDT)
        , ID        = modelDT$ID
        , label     = modelDT$label
        , fillcolor = modelDT$filledcolor
        , shape     = modelDT$shape
        , data      = modelDT$Feature
        , fontcolor = "black"
    )
    # The Split column might be numeric or character (e.g. if categorical features are used)
    # sometimes numeric <=0 splits are reported as <= 1.00000001800251e-35 or similar by lgb.model.dt.tree
    # replace these with "0"
    if (is.numeric(modelDT[["Split"]])) {
      modelDT[abs(Split) < .Machine$double.eps, Split := 0.0]
    }
    # for categorical features, LightGBM labels the splits as a single integer or
    # several integers separated by "||", e.g. "1" or "2||3||5"
    # if "rules" supplied, the integers are replaced by their corresponding factor level
    # to make the diagram easier to understand
    if (!is.null(rules)) {
      for (f in names(rules)) {
        modelDT[Feature == f & decision_type == "==", Split := unlist(lapply(
          Split,
          function(x) paste(names(rules[[f]])[as.numeric(unlist(strsplit(x, "||", fixed = TRUE)))], collapse = "\n")
        ))]
      }
    }
    # replace very long splits with a message as otherwise diagram will be very tall
    modelDT[nchar(Split) > 500L, Split := "Split too long to render"]
    # create the edges
    # define edgesDT to filter out leaf nodes
    edgesDT <- modelDT[Feature != "Leaf"]
    # create the edge data frame using edgesDT
    edges <- DiagrammeR::create_edge_df(
      from = match(rep(edgesDT[, ID], 2L), modelDT$ID),
      to = match(edgesDT[, c(Yes, No)], modelDT$ID),
      label = c(
        edgesDT[, paste(decision_type, Split)],
        rep("", nrow(edgesDT))
      ),
      # make the Missing edge bold
      style = c(
        edgesDT[, data.table::fifelse(Missing == Yes, "bold", "solid")],
        edgesDT[, data.table::fifelse(Missing == No, "bold", "solid")]
      ),
      rel = "leading_to"
    )
    # create the graph
    graph <- DiagrammeR::create_graph(
        nodes_df = nodes
        , edges_df = edges
        , attr_theme = NULL
    )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "graph"
        , attr = c("layout", "rankdir")
        , value = c("dot", "LR")
        )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "node"
        , attr = c("color", "style", "fontname")
        , value = c("DimGray", "filled", "Helvetica")
    )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "edge"
        , attr = c("color", "arrowsize", "arrowhead", "fontname")
        , value = c("DimGray", "1.5", "vee", "Helvetica")
    )
    # if 'render' is FALSE, return the graph object invisibly (without printing it)
    if (!render) {
      return(invisible(graph))
    } else {
      # if 'render' is TRUE, display the graph with specified width and height
      DiagrammeR::render_graph(graph, width = plot_width, height = plot_height)
    }
}
