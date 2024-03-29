

# Create a new column 'month_counter' using dplyr
df <- df %>%
  arrange(Date) %>%
  group_by(Month = cumsum(!duplicated(format(Date, "%Y-%m")))) %>%
  ungroup()

df <- df %>% 
  group_by(Home) %>%
  mutate(Ni = n())

df <- df %>% 
  group_by(Visitor) %>%
  mutate(Nj = n())

df$Result <- ifelse(df$`Score V` > df$`Score H`, 1, ifelse(df$`Score V` < df$`Score H`, 0, NA))

weight <- function(t, tmin = min(df$Month), tmax = max(df$Month)) ((1 + t - tmin)/(1 + tmax - tmin))**2

df$weight <- weight(df$Month)

eta <- function(p, P) ((1 + 0.1*P) / (p + 0.1*P))**0.602

neighbor_average <- function(wi, rk) sum(wi * rk) / sum(wi)

ri <- function(ri, eta, w, o_hut, o, lambda, alpha_i, Ni){
  return(ri - eta*(w*(o_hut - o)*o_hut*(1 - o_hut) + lambda/abs(Ni) * (ri - alpha_i)))
}

rj <- function(rj, eta, w, o_hut, o, lambda, alpha_j, Nj){
  return(rj - eta*(-w*(o_hut - o)*o_hut*(1 - o_hut) + lambda/abs(Nj) * (rj - alpha_j)))
}

pred <- function(ri, rj, g) 1/(1 + exp(rj - ri - gamma))



all.teams <- levels(as.factor(union(levels(as.factor(df$Home)),
                                    levels(as.factor(df$Visitor)))))

rating <- rep(0, times=length(all.teams))

ratings <- as.data.frame(rating, row.names = all.teams)

ratings$sum <- 0

ratings$sum_weights <- 0

ratings$neighbor_average <- 0

shuffled_data <- df[sample(1:nrow(df)),] 

for (idx in 1:dim(df)[1]){
  homeTeamName <- df[["Home"]][idx]
  awayTeamName <- df[["Visitor"]][idx]
  
  ratings[homeTeamName,]$sum <- ratings[homeTeamName,]$sum + df$weight[idx] * ratings[awayTeamName,]$rating
  ratings[homeTeamName,]$sum_weights <- ratings[homeTeamName,]$sum_weights + df$weight[idx]
  
  ratings[homeTeamName,]$neighbor_average <- ratings[homeTeamName,]$sum / ratings[homeTeamName,]$sum_weights
  
  print( ratings[homeTeamName,]$sum_weights)
}

learning_rate <- eta(p, P)

for (idx in 1:nrow(shuffled_data)){
  homeTeamName <- df[["Home"]][idx]
  awayTeamName <- df[["Visitor"]][idx]
  w <- df$weight[idx]
  Ni <- df$Ni[idx]
  Nj <- df$Nj[idx]
  o <- df$Result[idx]
  
  o_hut <-  1 / (1 + exp(ratings[homeTeamName,]$rating - ratings[awayTeamName,]$rating - 0.2))
#    pred(as.numeric(ratings[homeTeamName,]$rating), as.numeric(ratings[awayTeamName,]$rating), g = 0.2)
  
  ratings[homeTeamName,]$rating <- ri(ratings[homeTeamName,]$rating, learning_rate, w, o_hut, o, learning_rate, 
                                      ratings[homeTeamName,]$neighbor_average, Ni)
  
  ratings[awayTeamName,]$rating <-  ri(ratings[awayTeamName,]$rating, learning_rate, w, o_hut, o, learning_rate, 
                                       ratings[awayTeamName,]$neighbor_average, Nj)
}

  