##Read all the datasets 

nike1 <- read_xls("Nike1.xls", skip = 6)
attach(nike1)
names(nike1)
nike1
summary(nike1)

nike2 <- read_xls("Nike2.xls", skip = 6)
attach(nike2)
names(nike2)
nike2
summary(nike2)

nike3 <- read_xls("Nike3.xls", skip = 8)
attach(nike3)
names(nike3)
nike3
summary(nike3)

nike4 <- read_xls("Nike4.xls", skip = 4)
attach(nike4)
names(nike4)
nike4
summary(nike4)


nikedata <- read_xls("Nike data.xls", skip = 2)
attach(nikedata)
names(nikedata)
nikedata
summary(nikedata)


adidata <- read_xls("Adidas data.xls", skip = 3)
attach(adidata)
names(adidata)
adidata
summary(adidata)


##plot in nike1

plot(nike1$Time, nike1$Replies, xlab = "time in hr",ylab = "replies", type = "l")


##plot in nike2

plot(nike2$Week, nike2$Posts, xlab = "week", ylab = "no of posts", type = "l")


## plot in nike3


barplot(rbind(nike3$Nike,nike3$Adidas),names.arg=nike3$Metric,
        legend = c("Nike","Adidas"),ylab="Value",col=c("blue","red")
        ,border="red",las=2,beside = T)


##plot in nike4

plot(nike4$Year, nike4$Nike, type = "l", 
     col="blue", ylab="Revenue", ylim = c(7e09,3e10), xlab = "Year", lty=1)

lines(nike4$Year, nike4$Adidas, col="red", lty=2)


legend(x="topleft",legend = c("Nike", "Adidas"), col =c("blue", "red"), 
       lty = c(1,2))


##Plot Nike and Adidas followers data


plot(nikedata$Date,nikedata$Followers, type = "l",col="blue", 
     ylab="Followers", xlab = "Year", lty=1, ylim = c(1e6,10e6))

lines(adidata$Date, adidata$Followers, col="red", lty=2, 
      ylab="Followers", xlab = "Year")


legend(x="topleft",y=1e7,legend = c("Nike", "Adidas"), col =c("blue", "red"), 
       lty = c(1,2))


##Plot Nike and Adidas tweets data

plot(nikedata$Date,nikedata$Tweets, type = "l",col="blue", 
     ylab="Tweets", xlab = "Year", lty=1, ylim = c(1500,37000))

lines(adidata$Date, adidata$Tweets, col="red", lty=2, 
      ylab="Tweets", xlab = "Year")


legend(x="topleft",y=35000,legend = c("Nike", "Adidas"), col =c("blue", "red"), 
       lty = c(1,2))


###tweets and Revenue : Regression model for Nike

year_range<-nike4$Year[5:9]
Revenuenike_range <- nike4$Nike[5:9]
followersnike <- nikedata$Followers[6:2]

regtable <- data.frame(cbind(year_range,Revenuenike_range,followersnike))
regtable

linearMod <- lm(Revenuenike_range ~ followersnike, data=regtable)
summary(linearMod)

Revenuefitted <- predict(linearMod, list(regtable$followersnike))
Revenuefitted

plot(regtable$followersnike,regtable$Revenuenike_range, 
     type = "l",col="blue", 
     ylab="Nike Revenue", xlab = "Twitter Followers Nike", lty=1)

lines(regtable$followersnike, Revenuefitted, col="red", lty=2, 
      ylab="Tweets", xlab = "Followers")

legend(x="topleft",y=2e10,legend = c("Nike raw Revenue", "Fitted line"),
       col =c("blue", "red"),lty = c(1,2))

####Tweets and Revenue: Regression for Adidas

year_range<-nike4$Year[5:9]
Revenueadidas <- nike4$Adidas[5:9]
followersadi <- adidata$Followers[6:2]

regtable2 <- data.frame(cbind(year_range,Revenueadidas,followersadi))
regtable2

linearMod2 <- lm(Revenueadidas ~ followersadi,data=regtable2)
summary(linearMod2)

Revenuefitted2 <- predict(linearMod2, list(regtable2$followersadi))
Revenuefitted2

plot(regtable2$followersadi,regtable2$Revenueadidas, 
     type = "l",col="green", 
     ylab="Adidas Revenue", xlab = "Twitter Followers Adidas", lty=1)

lines(regtable2$followersadi, Revenuefitted2, col="red", lty=2, 
      ylab="Tweets", xlab = "Followers")

legend(x="topleft",y=1e10,legend = c("Adidas raw Revenue", "Fitted line"),
       col =c("green", "red"),lty = c(1,2))

