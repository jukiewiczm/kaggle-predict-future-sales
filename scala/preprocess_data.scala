// It's good to give more memory for execution, depending on your machine this should be set appropriately.
// I'm running standalone mode here so setting memory for driver is enough.
// Storage might take some space as well so set additional spark.local.dir to a disk with some space if needed.
// spark-shell --conf spark.driver.memory=20g -i preprocess_data.scala

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import scala.math

def readCsvWithHeaderInferSchema(path: String): DataFrame = {
  spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("quote", "\"")
    .option("escape", "\"")
    .csv(path)
}

def fillTestSet(
    testSet: DataFrame,
    testDateBlockNum: Long = 34,
    testDay: String = "01.11.2015"
): DataFrame = {
  testSet
    .withColumn("date", lit(testDay))
    .withColumn("date_block_num", lit(testDateBlockNum))
    .withColumn("item_price", lit(null).cast("double"))
    .withColumn("item_cnt_day", lit(null).cast("double"))
}

// Utility function for clipping values
val clipUdf = udf { (min: Long, max: Long, value: Long) =>
  math.max(min, math.min(max, value))
}

def expandMonthShopItem(monthlySet: DataFrame): DataFrame = {
  // Turn it on just for a while
  spark.conf.set("spark.sql.crossJoin.enabled", true)

  // Lookup for year and month for date_block_num
  val monthlyConstantValues =
    monthlySet.select($"date_block_num", $"year", $"month").distinct

  val monthShopItem =
    monthlySet.select($"date_block_num", $"shop_id", $"item_id").distinct.cache

  val monthShop = monthShopItem.select($"date_block_num", $"shop_id").distinct
  val monthItem = monthShopItem.select($"date_block_num", $"item_id").distinct

  val expandedSet = monthShop
    .join(monthItem, "date_block_num")
    .except(monthShopItem)
    .join(monthlyConstantValues, "date_block_num")
    .withColumn("item_cnt_month", lit(null).cast("int"))
    .withColumn("item_price", lit(null).cast("double"))

  spark.conf.set("spark.sql.crossJoin.enabled", false)

  expandedSet.unionByName(monthlySet)
}

def processData(
    salesSet: DataFrame,
    items: DataFrame,
    test_block_num: Int
): DataFrame = {

  val processedSet = salesSet
    .withColumn("date_parsed", to_date($"date", /*fmt=*/ "dd.MM.yyyy"))
    .withColumn("year", year($"date_parsed"))
    .withColumn("month", month($"date_parsed"))
    // The nulls are from test set only (last in the time order), so it's safe to keep them
    // I want to filter out negative prices here
    .filter($"item_price" >= 0 || isnull($"item_price"))
    .drop("date", "date_parsed")
    .groupBy($"date_block_num", $"shop_id", $"item_id")
    .agg(
      first($"year").as("year"),
      first($"month").as("month"),
      mean($"item_price").as("item_price"),
      sum($"item_cnt_day").as("item_cnt_month")
    )

  // Clarification on rowsBetween/rangeBetween before defining the actual windows:
  // rowsBetween means "x previous rows", which means that even if you take row -1,
  // it can be either a row from the same month, or a row from 5 months back, depending
  // on the partitioning strategy.
  // rangeBetween on the other hand means "x previous months" (values of the date_block_num)
  // This means that if you define it to -1 the value will be taken from one month back.
  // Depending on what you actually want to extract and the partitioning strategy,
  // using certain windows can be incorrect and case a leakage.
  // E.g. using rowsBetween is only correct when partitioning by shop/item, because
  // these combinations are unique within a month.
  // If I defined categoryShopMonthPreceding as a rowsBetween window, I would take
  // values from the same month since there are many rows with the same shop/category
  // combinations within one month.
  // rowsBetween/rangeBetween in the itemShopMonth* windows on the other hand is a matter
  // of preference and depends on what you'd like to extract. rowsBetween worked better for
  // me in this case.
  // Natually, many other windows of choice can be created.

  // Shop Item windows
  val itemShopMonthWindow = Window.partitionBy($"shop_id", $"item_id").orderBy($"date_block_num")

  val itemShopMonthPreceding = itemShopMonthWindow.rowsBetween(Window.unboundedPreceding, -1)
  val itemShopMonthPrecedingLastValue = itemShopMonthWindow.rowsBetween(-1, -1)
  val itemShopMonthPrecedingLast3Values = itemShopMonthWindow.rowsBetween(-3, -1)

  // Shop Item Category window
  val categoryShopMonthPreceding = Window
    .partitionBy($"shop_id", $"item_category_id")
    .orderBy($"date_block_num")
    .rangeBetween(Window.unboundedPreceding, -1)

  // Item windows
  val itemGeneric = Window.partitionBy($"item_id").orderBy($"date_block_num")

  val itemPreceding = itemGeneric.rangeBetween(Window.unboundedPreceding, -1)
  val itemPrecedingLastMonth = itemGeneric.rangeBetween(-1, -1)
  val itemPrecedingLastMonth_2 = itemGeneric.rangeBetween(-2, -2)
  val itemPrecedingLastMonth_3 = itemGeneric.rangeBetween(-3, -3)
  val itemPrecedingLast3Months = itemGeneric.rangeBetween(-3, -1)

  // This could be shortened but I decided to leave it like that for clarity.
  expandMonthShopItem(processedSet)
    .join(items.select($"item_id", $"item_category_id"), "item_id")
    .na.fill(0, Seq("item_cnt_month"))
    .withColumn("item_cnt_month", clipUdf(lit(0), lit(20), $"item_cnt_month"))

    .withColumn("item_shop_month_price_lag_1", lag($"item_price", 1).over(itemShopMonthWindow))
    .withColumn("item_shop_month_price_lag_2", lag($"item_price", 2).over(itemShopMonthWindow))
    .withColumn("item_shop_month_price_lag_3", lag($"item_price", 3).over(itemShopMonthWindow))

    .withColumn("last_item_shop_sale_interval", $"date_block_num" - lag($"date_block_num", 1).over(itemShopMonthWindow))
    .withColumn("last_item_sale_interval", $"date_block_num" - max($"date_block_num").over(itemPreceding))

    .withColumn("item_shop_month_price_mean", mean($"item_price").over(itemShopMonthPreceding))
    .withColumn("item_shop_month_price_max", max($"item_price").over(itemShopMonthPreceding))
    .withColumn("item_shop_month_cnt_mean", mean($"item_cnt_month").over(itemShopMonthPreceding))
    .withColumn("item_shop_month_cnt_max", max($"item_cnt_month").over(itemShopMonthPreceding))
    .withColumn("item_shop_times_previously", count(lit(1)).over(itemShopMonthPreceding))

    .withColumn("item_shop_last_value_price_mean", mean($"item_price").over(itemShopMonthPrecedingLastValue))
    .withColumn("item_shop_last_value_price_max", max($"item_price").over(itemShopMonthPrecedingLastValue))
    .withColumn("item_shop_last_value_cnt_mean", mean($"item_cnt_month").over(itemShopMonthPrecedingLastValue))
    .withColumn("item_shop_last_value_cnt_max", max($"item_cnt_month").over(itemShopMonthPrecedingLastValue))
    // This one basically tells you whether there was a sale at all at least once.
    .withColumn("item_shop_last_time_previously", count(lit(1)).over(itemShopMonthPrecedingLastValue))

    .withColumn("item_shop_last_3values_price_mean", mean($"item_price").over(itemShopMonthPrecedingLast3Values))
    .withColumn("item_shop_last_3values_price_max", max($"item_price").over(itemShopMonthPrecedingLast3Values))
    .withColumn("item_shop_last_3values_cnt_mean", mean($"item_cnt_month").over(itemShopMonthPrecedingLast3Values))
    .withColumn("item_shop_last_3values_cnt_max", max($"item_cnt_month").over(itemShopMonthPrecedingLast3Values))
    .withColumn("item_shop_last_3times_previously", count(lit(1)).over(itemShopMonthPrecedingLast3Values))

    .withColumn("item_cat_month_price_mean", mean($"item_price").over(categoryShopMonthPreceding))
    .withColumn("item_cat_month_price_max", max($"item_price").over(categoryShopMonthPreceding))
    .withColumn("item_cat_month_cnt_mean", mean($"item_cnt_month").over(categoryShopMonthPreceding))
    .withColumn("item_cat_month_cnt_max", max($"item_cnt_month").over(categoryShopMonthPreceding))
    .withColumn("item_cat_times_previously", count(lit(1)).over(categoryShopMonthPreceding))

    .withColumn("item_shop_month_cnt_lag_1", lag($"item_cnt_month", 1).over(itemShopMonthWindow))
    .withColumn("item_shop_month_cnt_lag_2", lag($"item_cnt_month", 2).over(itemShopMonthWindow))
    .withColumn("item_shop_month_cnt_lag_3", lag($"item_cnt_month", 3).over(itemShopMonthWindow))

    .withColumn("item_count_sold_previously", count(lit(1)).over(itemPreceding))
    .withColumn("item_mean_count_previously", mean($"item_cnt_month").over(itemPreceding))
    .withColumn("item_max_count_previously", max($"item_cnt_month").over(itemPreceding))

    .withColumn("last_item_price_lag_1", mean($"item_price").over(itemPrecedingLastMonth))
    .withColumn("last_item_price_lag_2", mean($"item_price").over(itemPrecedingLastMonth_2))
    .withColumn("last_item_price_lag_3", mean($"item_price").over(itemPrecedingLastMonth_3))

    .withColumn("item_count_sold_last_month", count(lit(1)).over(itemPrecedingLastMonth))
    .withColumn("item_mean_count_last_month", mean($"item_cnt_month").over(itemPrecedingLastMonth))
    .withColumn("item_max_count_last_month", max($"item_cnt_month").over(itemPrecedingLastMonth))

    .withColumn("item_count_sold_last_3month", count(lit(1)).over(itemPrecedingLast3Months))
    .withColumn("item_mean_count_last_3month", mean($"item_cnt_month").over(itemPrecedingLast3Months))
    .withColumn("item_max_count_last_3month", max($"item_cnt_month").over(itemPrecedingLast3Months))
    .drop($"item_price")
}

def processDatasets(
    sets: Seq[DataFrame],
    items: DataFrame,
    test_block_num: Int
): (DataFrame, DataFrame) = {
  val setsUnion = sets.reduce(_.unionByName(_))
  val unionProcessed = processData(setsUnion, items, test_block_num).cache()

  val trainProcessed =
    unionProcessed.filter($"date_block_num" =!= test_block_num)
  val testProcessed =
    unionProcessed.filter($"date_block_num" === test_block_num)

  (trainProcessed, testProcessed)
}

val rootPath = "kaggle-predict-future-sales"
val trainPath = s"$rootPath/data/sales_train.csv"
val testPath = s"$rootPath/data/test.csv"
val itemsPath = s"$rootPath/data/items.csv"
val itemsCategoriesPath = s"$rootPath/data/item_categories.csv"
val shopsPath = s"$rootPath/data/shops.csv"

val trainSet = readCsvWithHeaderInferSchema(trainPath)
val testSet = readCsvWithHeaderInferSchema(testPath)
val items = readCsvWithHeaderInferSchema(itemsPath)
val itemsCategories = readCsvWithHeaderInferSchema(itemsCategoriesPath)
val shops = readCsvWithHeaderInferSchema(shopsPath)

val testSetFilled = fillTestSet(testSet)

// Prepare whole train set and test set
val processedTrueRoot = s"$rootPath/data/processed_full"
val (trainProcessedTrue, testProcessedTrue) =
  processDatasets(Seq(trainSet, testSetFilled.drop("ID")), items, 34)

trainProcessedTrue.write.parquet(s"$processedTrueRoot/train.parquet")
testProcessedTrue
  .join(
    testSet.select($"ID", $"item_id", $"shop_id"),
    Seq("item_id", "shop_id")
  )
  .write
  .parquet(s"$processedTrueRoot/test.parquet")

// Prepare train set with validation set
val processedValidRoot = s"$rootPath/data/processed_validation"
val (trainProcessedVal, testProcessedVal) =
  processDatasets(Seq(trainSet), items, 33)

trainProcessedVal.write.parquet(s"$processedValidRoot/train.parquet")
testProcessedVal.write.parquet(s"$processedValidRoot/test.parquet")

System.exit(0)
