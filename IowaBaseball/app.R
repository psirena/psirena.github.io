#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(randomForest)
library(DT)

FilterTM <- read.csv("FilterTM.csv")
FilterTM2 <- read.csv("FilterTM2.csv")
FilterTM3 <- read.csv("FilterTM3.csv")


# {r Pitcher Cluster Stats Function}
PitcherClusterStats <- function(pitcher) {
    Pdata <- FilterTM %>% filter(Pitcher==pitcher) %>% group_by(Pitcher,PitcherThrows) %>% summarize(n = n()) %>% filter(n == max(n))
    if (Pdata$PitcherThrows == "Right") {
        ClusterStats <- FilterTM2 %>% filter(Pitcher==pitcher) %>% group_by(Cluster, Group, PitcherThrows) %>% filter(n() >= 10) %>%
            summarize('# Thrown' = n(), 'Zone %' = round(sum(InZone=="Yes")/n()*100,1),
                      'Zone Swing %' = round(sum(InZone=="Yes" & Swing=="Yes")/sum(InZone=="Yes")*100,1),
                      'Zone Contact %' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                   sum(InZone=="Yes" & Swing=="Yes")*100,1),
                      'Chase %' = round(sum(InZone=="No" & Swing=="Yes")/sum(InZone=="No")*100,1),
                      'Chase Contact %' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                    sum(InZone=="No" & Swing=="Yes")*100,1))
        Krate <- sum(filter(FilterTM,Pitcher==pitcher)$KorBB=="Strikeout")/sum(filter(FilterTM,Pitcher==pitcher)$PitchofPA==1)
        BBrate <- sum(filter(FilterTM,Pitcher==pitcher)$KorBB=="Walk")/sum(filter(FilterTM,Pitcher==pitcher)$PitchofPA==1)
        KorBBrate <- Krate+BBrate
        ClusterStats$Krate <- Krate/KorBBrate
        return(ClusterStats)
    }
    else {
        ClusterStats <- FilterTM3 %>% filter(Pitcher==pitcher) %>% group_by(Cluster, Group, PitcherThrows) %>% filter(n() >= 10) %>%
            summarize('# Thrown' = n(), 'Zone %' = round(sum(InZone=="Yes")/n()*100,1),
                      'Zone Swing %' = round(sum(InZone=="Yes" & Swing=="Yes")/sum(InZone=="Yes")*100,1),
                      'Zone Contact %' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA")/sum(InZone=="Yes" & Swing=="Yes")*100,1),
                      'Chase %' = round(sum(InZone=="No" & Swing=="Yes")/sum(InZone=="No")*100,1),
                      'Chase Contact %' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA")/sum(InZone=="No" & Swing=="Yes")*100,1))
        Krate <- sum(filter(FilterTM,Pitcher==pitcher)$KorBB=="Strikeout")/sum(filter(FilterTM,Pitcher==pitcher)$PitchofPA==1)
        BBrate <- sum(filter(FilterTM,Pitcher==pitcher)$KorBB=="Walk")/sum(filter(FilterTM,Pitcher==pitcher)$PitchofPA==1)
        KorBBrate <- Krate+BBrate
        ClusterStats$Krate <- Krate/KorBBrate
        return(ClusterStats)
    }
}


# {r Hitter Cluster Stats Function}
HitterClusterStats <- function(hitter,Left_Right) {
    # Choose hitter and whether they are facing left or right hander
    vsRHP <- FilterTM2 %>% filter(Batter==hitter) %>% group_by(Group) %>% filter(n() >= 10) %>%
        summarize('# Seen' = n(), 'Hitter Zone %' = round(sum(InZone=="Yes",na.rm=TRUE)/n()*100,1),
                  'Hitter Zone Swing %' = round(sum(InZone=="Yes" & Swing=="Yes",na.rm=TRUE)/sum(InZone=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Zone Contact %' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                      sum(InZone=="Yes"&Swing=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Chase %' = round(sum(InZone=="No" & Swing=="Yes",na.rm=TRUE)/sum(InZone=="No",na.rm=TRUE)*100,1),
                  'Hitter Chase Contact %' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                       sum(InZone=="No" & Swing=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Average EV' = round(mean(ExitSpeed,na.rm=TRUE),1),
                  'Hitter EV SD' = round(sd(ExitSpeed,na.rm=TRUE),1),
                  'Hitter Average LA' = round(mean(Angle,na.rm=TRUE),1),
                  'Hitter LA SD' = round(sd(Angle,na.rm=TRUE),1),
                  'Hitter Average Direction' = round(mean(Direction,na.rm=TRUE),1),
                  'Hitter Direction SD' = round(sd(Direction,na.rm=TRUE),1))
    Krate <- sum(filter(FilterTM2,Batter==hitter)$KorBB=="Strikeout")/sum(filter(FilterTM2,Batter==hitter)$PitchofPA==1)
    BBrate <- sum(filter(FilterTM2,Batter==hitter)$KorBB=="Walk")/sum(filter(FilterTM2,Batter==hitter)$PitchofPA==1)
    KorBBrate <- Krate+BBrate
    vsRHP$`Hitter K Rate` <- Krate/KorBBrate
    
    vsLHP <- FilterTM3 %>% filter(Batter==hitter) %>% group_by(Group) %>% filter(n() >= 10) %>%
        summarize('# Seen' = n(), 'Hitter Zone %' = round(sum(InZone=="Yes",na.rm=TRUE)/n()*100,1),
                  'Hitter Zone Swing %' = round(sum(InZone=="Yes" & Swing=="Yes",na.rm=TRUE)/sum(InZone=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Zone Contact %' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                      sum(InZone=="Yes"&Swing=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Chase %' = round(sum(InZone=="No" & Swing=="Yes")/sum(InZone=="No",na.rm=TRUE)*100,1),
                  'Hitter Chase Contact %' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/
                                                       sum(InZone=="No" & Swing=="Yes",na.rm=TRUE)*100,1),
                  'Hitter Average EV' = round(mean(ExitSpeed,na.rm=TRUE),1),
                  'Hitter EV SD' = round(sd(ExitSpeed,na.rm=TRUE),1),
                  'Hitter Average LA' = round(mean(Angle,na.rm=TRUE),1),
                  'Hitter LA SD' = round(sd(Angle,na.rm=TRUE),1),
                  'Hitter Average Direction' = round(mean(Direction,na.rm=TRUE),1),
                  'Hitter Direction SD' = round(sd(Direction,na.rm=TRUE),1))
    Krate <- sum(filter(FilterTM3,Batter==hitter)$KorBB=="Strikeout")/sum(filter(FilterTM3,Batter==hitter)$PitchofPA==1)
    BBrate <- sum(filter(FilterTM3,Batter==hitter)$KorBB=="Walk")/sum(filter(FilterTM3,Batter==hitter)$PitchofPA==1)
    KorBBrate <- Krate+BBrate
    vsLHP$`Hitter K Rate` <- Krate/KorBBrate
    
    if (Left_Right == "Left") {
        return(vsLHP)
    }
    else {
        return(vsRHP)
    }
}


# {r Combined Cluster Stats Function}
ClusterStats <- function(pitcher,hitter) {
    PitcherStats <- PitcherClusterStats(pitcher)
    BatterStats <- HitterClusterStats(hitter,levels(as.factor(PitcherStats$PitcherThrows)))
    CombinedStats <- left_join(PitcherStats,BatterStats,"Group")
    CombinedStats$`# Seen` <- as.numeric(CombinedStats$`# Seen`)
    
    for (i in 1:nrow(CombinedStats)) {
        for (j in 1:ncol(CombinedStats)) {
            if (is.na(CombinedStats[i,j])) {
                CombinedStats[i,j] <- mean(as.data.frame(CombinedStats[,j])[,1],na.rm=TRUE)
            }
        }
    }
    
    CombinedStats <- CombinedStats %>% 
        transmute(Cluster = Cluster, Group = Group, PitcherThrows = PitcherThrows,
                  'Usage' = mean(c(`# Thrown`,`# Seen`)), 'Zone' = mean(c(`Zone %`,`Hitter Zone %`)),
                  'ZoneSwing' = mean(c(`Zone Swing %`,`Hitter Zone Swing %`)), 
                  'ZoneContact' = mean(c(`Zone Contact %`,`Hitter Zone Contact %`)), 'Chase' = mean(c(`Chase %`,`Hitter Chase %`)),
                  'ChaseContact' = mean(c(`Chase Contact %`,`Hitter Chase Contact %`)),
                  'Hitter Average EV' = `Hitter Average EV`, 'Hitter EV SD' = `Hitter EV SD`,
                  'Hitter Average LA' = `Hitter Average LA`, 'Hitter LA SD' = `Hitter LA SD`,
                  'Hitter Average Direction' = `Hitter Average Direction`, 'Hitter Direction SD' = `Hitter Direction SD`,
                  'K Rate' = mean(c(Krate,`Hitter K Rate`))) 
    
    PitchCount <- sum(CombinedStats$Usage)
    
    CombinedStats <- mutate(CombinedStats,Usage=Usage/PitchCount)
    CombinedStats$Usage <- (CombinedStats$Usage-min(CombinedStats$Usage))/(max(CombinedStats$Usage)-min(CombinedStats$Usage))
    
    return(CombinedStats)
}


# {r Training Sets}
# RHH

TrainRHH <- FilterTM2 %>% group_by(Cluster,Group) %>%
    transmute(Cluster = Cluster, Group = Group, PitcherThrows = PitcherThrows,
              'Usage' = n() / nrow(FilterTM2), 'Zone' = round(sum(InZone=="Yes")/n()*100,1),
              'ZoneSwing' = round(sum(InZone=="Yes" & Swing=="Yes")/sum(InZone=="Yes")*100,1),
              'ZoneContact' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/sum(InZone=="Yes" & Swing=="Yes")*100,1),
              'Chase' = round(sum(InZone=="No" & Swing=="Yes")/sum(InZone=="No")*100,1),
              'ChaseContact' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA")/sum(InZone=="No" & Swing=="Yes")*100,1),
              'Hitter In-Play over 57?' = ifelse(round(sum(PitchCall=="InPlay")/sum(PitchofPA==1)*100,1)>=57,1,0)) %>% 
    unique() %>% na.omit()

for (i in 1:nrow(TrainRHH)) {
    for (j in 6:ncol(TrainRHH)) {
        if (is.na(TrainRHH[i,j])) {
            TrainRHH[i,j] <- mean(as.data.frame(TrainRHH[,j])[,1],na.rm=TRUE)
        }
    }
}

TrainRHH$`Hitter In-Play over 57?` <- as.factor(ifelse(TrainRHH$`Hitter In-Play over 57?` >= .5,1,0))

# LHH

TrainLHH <- FilterTM3 %>% group_by(Cluster,Group) %>%
    transmute(Cluster = Cluster, Group = Group, PitcherThrows = PitcherThrows,
              'Usage' = n() / nrow(FilterTM2), 'Zone' = round(sum(InZone=="Yes")/n()*100,1),
              'ZoneSwing' = round(sum(InZone=="Yes" & Swing=="Yes")/sum(InZone=="Yes")*100,1),
              'ZoneContact' = round(sum(InZone=="Yes" & Swing=="Yes" & Contact!="NA",na.rm=TRUE)/sum(InZone=="Yes" & Swing=="Yes")*100,1),
              'Chase' = round(sum(InZone=="No" & Swing=="Yes")/sum(InZone=="No")*100,1),
              'ChaseContact' = round(sum(InZone=="No" & Swing=="Yes" & Contact!="NA")/sum(InZone=="No" & Swing=="Yes")*100,1),
              'Hitter In-Play over 57?' = ifelse(round(sum(PitchCall=="InPlay")/sum(PitchofPA==1)*100,1)>=57,1,0)) %>% 
    unique() %>% na.omit()

for (i in 1:nrow(TrainLHH)) {
    for (j in 6:ncol(TrainLHH)) {
        if (is.na(TrainLHH[i,j])) {
            TrainLHH[i,j] <- mean(as.data.frame(TrainLHH[,j])[,1],na.rm=TRUE)
        }
    }
}

TrainLHH$`Hitter In-Play over 57?` <- as.factor(ifelse(TrainLHH$`Hitter In-Play over 57?` >= .5,1,0))

# Total

Train <- rbind(TrainRHH,TrainLHH)
Train$Usage <- (Train$Usage-min(Train$Usage))/(max(Train$Usage)-min(Train$Usage))


# {r In-Play Model}
set.seed(2021)
InPlayModel <- randomForest(`Hitter In-Play over 57?`~.,data=Train[,3:10])
InPlayModel


# {r Sample Function}
SamplePA <- function(df,n=4) {
    return(df[sample(nrow(df),n,replace=TRUE),])
}


# {r Outcome Model}
OutcomeData <- FilterTM %>% 
    mutate(woba_num =
               ifelse(PlayResult=="Single",1,
                      ifelse(PlayResult== "Double",2,
                             ifelse(PlayResult == "Triple",3,
                                    ifelse(PlayResult=="HomeRun",4,0))))) %>%
    filter(PlayResult != "Undefined") %>%  droplevels()

cols = c("ExitSpeed", "Angle", "Direction", "woba_num")
OutcomeData = na.omit(OutcomeData[,cols])
names(OutcomeData) <- c("EV","LA","Direction","woba_num")
OutcomeData$woba_num <- as.factor(OutcomeData$woba_num)

set.seed(2021)

Outcome <- randomForest(woba_num~.,importance=TRUE,data=OutcomeData)
Outcome


# {r Outcome Model}
SimulatedStats <- function(pitcher,hitter,num=300) {
    df <- ClusterStats(pitcher,hitter)
    PAs <- SamplePA(df,n=num)
    PAs$`Hitter In-Play over 57?` <- predict(InPlayModel,PAs[,3:9])
    PAs$EV <- 0
    PAs$LA <- 0
    PAs$Direction <- 0
    PAs$woba_num <- as.factor(0)
    PAs$KorBB <- "NA"
    levels(PAs$woba_num) <- c("0","1","2","3","4")
    
    for (i in 1:nrow(PAs)) {
        PAs[i,18] <- ifelse(PAs[i,17]==1,rnorm(1,as.numeric(PAs[i,10]),as.numeric(PAs[i,11])),NA)
        PAs[i,19] <- ifelse(PAs[i,17]==1,rnorm(1,as.numeric(PAs[i,12]),as.numeric(PAs[i,13])),NA)
        PAs[i,20] <- ifelse(PAs[i,17]==1,rnorm(1,as.numeric(PAs[i,14]),as.numeric(PAs[i,15])),NA)
        
        if (!is.na(PAs[i,18])) {
            PAs[i,21] <- predict(Outcome,PAs[i,18:20])
        }
        else {
            if (!is.na(PAs[[i,16]])) {
                PAs[i,22] <- sample(c("Strikeout","Walk"),size=1,prob=c(PAs[[i,16]],1-PAs[[i,16]]))
            }
            else {
                PAs[i,22] <- sample(c("Strikeout","Walk"),size=1,prob=c(2/3,1/3))
            }
        }
    }
    
    return(PAs)
}


# {r Get Simulation Stats}
SimulatedBoxScore <- function(pitcher,hitter,number=300) {
    df <- SimulatedStats(pitcher,hitter,num=number)
    
    BoxScore <- df %>% ungroup() %>%
        summarize(Batter = hitter, PA = number, AB = sum(KorBB!="Walk"), 
                  H = sum(woba_num!=0), '1B' = sum(woba_num==1), '2B' = sum(woba_num==2), '3B' = sum(woba_num==3), HR = sum(woba_num==4),
                  K = sum(KorBB=="Strikeout"), BB = sum(KorBB=="Walk"), 'Hard-hit Balls' = sum(EV>=95,na.rm=TRUE))
    
    return(BoxScore)
}


# {r Multiple Simulations}
Simulations <- function(pitcher,hitter,numberPA=300) {
    df <- SimulatedBoxScore(pitcher,hitter,number=numberPA)
    
    Stats <- df %>% mutate('Hard-hit Rate' = round(`Hard-hit Balls` / (AB-K) * 100, 1),
                           'AVG' = round(H / AB, 3), 'OBP' = round((H+BB) / PA, 3), 'SLG' = round((`1B`+2*`2B`+3*`3B`+4*HR) / AB, 3),
                           'OPS' = OBP + SLG)
    
    return(Stats)
}


# {r Pitchers and Hitters By Team}
PitcherTeams <- FilterTM %>% filter(Semester=="Spring 2020" & Competition != "Exhibition" & Competition != "Intrasquad") %>% 
    select(PitcherTeam,Pitcher) %>% unique()
BatterTeams <- FilterTM %>% filter(Semester=="Spring 2020" & Competition != "Exhibition" & Competition != "Intrasquad") %>% 
    select(BatterTeam,Batter) %>% unique()

# {r Conferences}
HomeConf <- FilterTM %>% filter(Semester=="Spring 2020") %>% select(HomeTeam,HomeTeamConf) %>% unique()
HomeConf1 <- HomeConf
AwayConf <- FilterTM %>% filter(Semester=="Spring 2020") %>% select(AwayTeam,AwayTeamConf) %>% unique()
AwayConf1 <- AwayConf

names(HomeConf)[1] <- "PitcherTeam"
names(HomeConf1)[1] <- "BatterTeam"
names(AwayConf)[1] <- "PitcherTeam"
names(AwayConf1)[1] <- "BatterTeam"


# {r Adding Conferences to Pitchers and Hitters}
PitcherTeams <- left_join(PitcherTeams,HomeConf,"PitcherTeam")
PitcherTeams <- left_join(PitcherTeams,AwayConf,"PitcherTeam")

BatterTeams <- left_join(BatterTeams,HomeConf1,"BatterTeam")
BatterTeams <- left_join(BatterTeams,AwayConf1,"BatterTeam")

PitcherTeams$Conf <- ifelse(is.na(PitcherTeams$AwayTeamConf),PitcherTeams$HomeTeamConf,PitcherTeams$AwayTeamConf)
PitcherTeams <- PitcherTeams %>% select(Conf,PitcherTeam,Pitcher)

BatterTeams$Conf <- ifelse(is.na(BatterTeams$AwayTeamConf),BatterTeams$HomeTeamConf,BatterTeams$AwayTeamConf)
BatterTeams <- BatterTeams %>% select(Conf,BatterTeam,Batter)

nPitches <- FilterTM %>% group_by(BatterTeam,Batter) %>% summarize(n=n()) 

# {r Simulate Stats for Entire Team}
LineupSim <- function(pitcher,team) {
    LineupStats <- Simulations(pitcher,team$Batter[1])
    for (i in 2:nrow(team)) {
        LineupStats <- rbind(LineupStats,Simulations(pitcher,team$Batter[i]))
    }
    return(LineupStats)
}

# Define UI for application 

ui <- fluidPage(

    # Application title
    titlePanel("Simulated Stats"),

    sidebarPanel(
        selectInput(
            inputId = "PitcherConference", 
            label = "Select Pitcher Conference", 
            choices = sort(PitcherTeams$Conf)),
        
        selectInput(
            inputId = "PitcherTeam",
            label = "Select Pitcher Team",
            choices = sort(PitcherTeams$PitcherTeam)),
        
        selectInput(
            inputId = "Pitcher",
            label = "Select Pitcher",
            choices = sort(PitcherTeams$Pitcher)),
 
        selectInput(
            inputId = "OpposingTeamConference",
            label = "Select Opposing Team Conference",
            choices = sort(BatterTeams$Conf)),
        
        selectInput(
            inputId = "OpposingTeam",
            label = "Select Opposing Team",
            choices = sort(BatterTeams$BatterTeam)),
        
        actionButton("Simulate","Simulate Plate Appearances"),
        
        img(src = "Tigerhawk.png", 
            style = "display: block; margin-left: auto; 
                     margin-right: auto;", 
            height = 150, 
            width = 150)
    ),
    
    mainPanel(
        
        dataTableOutput("Simulations")
        
    )
)

# Define server logic 

server <- function(input, output, session) {
    
    observeEvent(input$PitcherConference,
                 updateSelectInput(session,"PitcherTeam","Select Pitcher Team",choices=
                                       sort(unique(PitcherTeams$PitcherTeam[PitcherTeams$Conf==input$PitcherConference]))))
    
    observeEvent(input$PitcherTeam,
                 updateSelectInput(session,"Pitcher","Select Pitcher",choices=
                                       sort(unique(PitcherTeams$Pitcher[PitcherTeams$PitcherTeam==input$PitcherTeam]))))
    
    observeEvent(input$OpposingTeamConference,
                 updateSelectInput(session,"OpposingTeam","Select Opposing Team",choices=
                                       sort(unique(BatterTeams$BatterTeam[BatterTeams$Conf==input$OpposingTeamConference]))))
    
    output$Simulations <- renderDataTable({
        req(input$Pitcher)
        req(input$OpposingTeam)
        req(input$Simulate)
        Line <- LineupSim(input$Pitcher,input$OpposingTeam)
        datatable(Line)
    })
    
}

# Run the application 

shinyApp(ui = ui, server = server)
