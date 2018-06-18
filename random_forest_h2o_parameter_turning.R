

# Sử dụng bộ dữ liệu AmesHousing: 
rm(list = ls())
library(MASS)
data("Boston")

# Chuẩn bị dữ liệu: 

library(caret)
set.seed(1)
id <- createDataPartition(y = Boston$medv, p = 0.7, list = FALSE)

train <- Boston[id, ]
test <- Boston[-id, ]

#-----------------------
# Default Random Forest 
#-----------------------

library(randomForest)
set.seed(123)
m1 <- randomForest(medv ~ ., data = train)

# Viết hàm biểu diễn OOB MSE theo số lượng trees và chỉ ra 
# tree tối ưu được biểu diễn bằng điểm màu đỏ: 

library(tidyverse)

my_OOB_plot <- function(RF_model) {
  n <- RF_model$mse %>% length()
  optimal_trees <- which.min(RF_model$mse)
  sub_tit <- paste0("Optimal Tree (Red Point): ", optimal_trees) 
  error <- data.frame(Error = RF_model$mse, Trees = 1:n)
  error %>% 
    ggplot(aes(Trees, Error)) + 
    geom_line() + 
    geom_point(data = error %>% slice(which.min(Error)), color = "red", size = 2) + 
    labs(y = "OOB Error", x = "Number of Trees", 
         title = "The Relationhip Between OOB Error and Number of Trees", 
         subtitle = sub_tit) + 
    theme_minimal()
  
  
}

# Số Trees tối ưu: 
my_OOB_plot(m1)

#-----------------------------------------
#  Sử dụng ranger để tiết kiệm thời gian
#-----------------------------------------

system.time(boston_randomForest <- randomForest(medv ~ ., data = train))

library(ranger)
system.time(boston_ranger <- ranger(medv ~ ., data = train))


#------------------------------------------------
#   Chơi h2o tinh chỉnh và chọn tham số tối ưu
#------------------------------------------------ 

# Thiết lập các lựa chọn để sử dụng h2o cho Random Forest: 

library(h2o)
h2o.no_progress()
h2o.init(max_mem_size = "12g")

# Chuẩn bị dữ liệu: 
y <- "medv"
x <- setdiff(names(train), y)

# Chuyển hóa về h2o object: 
train.h2o <- as.h2o(train)

# Thiết lập một loạt các giá trị của tham số: 

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 100),
                       mtries = seq(2, 13, by = 2),
                       sample_rate = c(.55, .632, .70, .80))

# Nghĩa là sẽ có ít nhất 120 mô hình Random Forest được huấn luyện: 
hyper_grid.h2o %>% sapply(length) %>% prod()

# Huấn luyện 120 mô hình RF (nên thời gian có thể hơi lâu). Hệ thống
# sẽ sử dụng xấp xỉ từ 9 đến 11  Gi RAM nên nếu máy tính nào có RAM
# yếu thì có thể không chơi được game này: 

grid <- h2o.grid(algorithm = "randomForest",
                 grid_id = "rf_grid",
                 x = x, 
                 y = y, 
                 training_frame = train.h2o,
                 seed = 1, 
                 hyper_params = hyper_grid.h2o,
                 search_criteria = list(strategy = "Cartesian"))

# Sắp xếp kết quả và cũng là mô hình có tham số tối ưu theo, ví dụ, 
# tiêu chí lựa chọn là MSE: 
grid_perf <- h2o.getGrid(grid_id = "rf_grid",
                         sort_by = "mse", 
                         decreasing = FALSE)

# In ra kết quả và mô hình có tham số tối ưu là 
# mtry = 6, ntree = 250 và sample rate = 0.8: 
print(grid_perf)

# Hàm lấy ra mô hình có tham số tối ưu trong số một loạt
# các ứng viên về tham số mà ta đã thiết lập: 

get_best_model <- function(h2o_object_grid) {
  h2o.getModel(h2o_object_grid@model_ids[[1]]) %>% return()
  
}

# Sử dụng hàm: 

best_model <- get_best_model(grid_perf)

# Sử dụng mô hình đã huấn luyện được để thực hiện 
# dự báo trên bộ dữ liệu test. Viết hàm dự báo: 

du_bao <- function(your_model, test_df) {
  predict(your_model, test %>% as.h2o()) %>% 
    as.vector() %>% 
    return()
}

# Sử dụng hàm (xem 6 kết quả dự báo đầu tiên): 
pred_rf <- du_bao(best_model, test)
pred_rf %>% head()

# So với giá trị thực: 
test$medv %>% head()

#------------------------
#    So sánh với OLS
#------------------------

my_ols <- lm(medv ~., data = train)
pred_ols <- predict(my_ols, test)
pred_ols %>% head()


# Viết hàm tính R2: 

my_rsquared <- function(pred, actual) {
  r2 <- cor(pred, actual)
  return(r2*r2)
}

# So sánh R2 của Random Forest và OLS: 
my_rsquared(pred_rf, test$medv)
my_rsquared(pred_ols, test$medv)

# So với RF mặc định không tinh chỉnh: 

my_rsquared(predict(boston_randomForest, test), test$medv)
my_rsquared(predict(boston_ranger, test) %>% .[[1]], test$medv)


#-----------------------------
#  Tinh chỉnh theo kiểu khác
#-----------------------------

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 100),
                       mtries = seq(2, 13, by = 2),
                       sample_rate = c(.55, .632, .70, .80), 
                       max_depth = seq(20, 40, by = 5),
                       min_rows  = seq(1, 5, by = 2),
                       nbins = seq(10, 30, by = 5))


hyper_grid.h2o %>% sapply(length) %>% prod()


search_criteria <- list(strategy = "RandomDiscrete",
                        stopping_metric = "mse",
                        stopping_tolerance = 0.005,
                        stopping_rounds = 10,
                        max_runtime_secs = 30*60)


system.time(
  random_grid <- h2o.grid(algorithm = "randomForest",
                          grid_id = "rf_grid2",
                          x = x, 
                          y = y, 
                          training_frame = train.h2o,
                          hyper_params = hyper_grid.h2o,
                          search_criteria = search_criteria)
)


grid_perf2 <- h2o.getGrid(grid_id = "rf_grid2", 
                          sort_by = "mse", 
                          decreasing = FALSE)



# R2 của mô hình: 

best_model2 <- get_best_model(grid_perf2)
my_rsquared(du_bao(best_model2, test), test$medv)














