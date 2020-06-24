import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Baseline model with added dropout
class SimpleModel2(nn.Module):

    def __init__(self, person_count, person_feature_size, tagret_size, dropout_p = 0):
        super(SimpleModel2, self).__init__()

        # initial dense layer with shared weights
        self.linear1 = nn.Linear(person_feature_size, person_feature_size)
        self.linear2 = nn.Linear(person_feature_size, person_feature_size)
        self.linear3 = nn.Linear(person_feature_size, person_feature_size)
        self.reLu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # LSTM
        self.lstm = nn.LSTM(person_feature_size, person_feature_size, batch_first=False )
        # (seq, batch, input_size)
        self.hidden = torch.randn(1, person_count, person_feature_size).cuda()
        # hidden state distinct for each person
        self.cell_state = torch.randn(1, person_count, person_feature_size).cuda()

        self.out = nn.Linear(person_feature_size, tagret_size)

    def forward(self, personData):

        result = []
        hidden_individual = []

        p = self.linear1(personData)
        p = self.reLu(p)
        p = self.dropout(p)
        p = self.linear2(p)
        p = self.reLu(p)
        p = self.dropout(p)
        p = self.linear3(p)
        p = self.reLu(p)
        p = self.dropout(p)

        p, (hidden, cell_state) = self.lstm(
            p, (self.hidden, self.cell_state))

        self.hidden = hidden.detach()
        self.cell_state = cell_state.detach()

        result = self.out(p)
        result = self.reLu(result)
        return result

# Team model with added dropout, this model was used to produce result
class TeamModel2(nn.Module):

    def __init__(self, person_count, person_feature_size, tagret_size, dropout_p = 0):
        super(TeamModel2, self).__init__()

        # initial dense layer with shared weights
        self.linear1 = nn.Linear(person_feature_size, person_feature_size)
        self.linear2 = nn.Linear(person_feature_size, person_feature_size)
        self.linear3 = nn.Linear(person_feature_size, person_feature_size)
        self.reLu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # personal LSTM
        self.personal_lstm = nn.LSTM(person_feature_size, person_feature_size, batch_first=False)
        # (seq, batch, input_size)
        self.hidden = torch.randn(1, person_count, person_feature_size).cuda()
        # hidden state distinct for each person
        self.cell_state = torch.randn(1, person_count, person_feature_size).cuda()

        # team LSTM
        self.team_pool = nn.Linear(person_feature_size * 3, person_feature_size)
        self.team_lstm = nn.LSTM(person_feature_size, person_feature_size, batch_first=False)
        self.team_hidden = torch.randn(1, 1, person_feature_size).cuda()
        self.team_cell_state = torch.randn(1, 1, person_feature_size).cuda()

        # combine team output with personal output
        self.combine = nn.Linear(person_feature_size * 2, person_feature_size)
        
        self.out = nn.Linear(person_feature_size, tagret_size)

    def forward(self, personData):

        result = []
        hidden_individual = []

        p = self.linear1(personData)
        p = self.reLu(p)
        p = self.dropout(p)
        p = self.linear2(p)
        p = self.reLu(p)
        p = self.dropout(p)
        p = self.linear3(p)
        p = self.reLu(p)
        p = self.dropout(p)

        personal_lstm_out, (hidden, cell_state) = self.personal_lstm(
            p, (self.hidden, self.cell_state))
        self.hidden = hidden.detach()
        self.cell_state = cell_state.detach()

        # (frame, person, features)
        shape = personal_lstm_out.size() 
        team_pooled = self.team_pool(torch.reshape(personal_lstm_out, (shape[0], 1, shape[1] * shape[2])))
        team_pooled = self.reLu(team_pooled)

        team_lstm_out, (team_hidden, team_cell_state) = self.team_lstm(
            team_pooled, (self.team_hidden, self.team_cell_state))
        self.team_hidden = team_hidden.detach()
        self.team_cell_state = team_cell_state.detach()

        # copy team lstm for each person so it can be combined with personal ones
        team_lstm_out_by_person = torch.cat([team_lstm_out] * 3, 1)

        # concatenate features comming from personal model and team model
        combine_in = torch.cat([personal_lstm_out, team_lstm_out_by_person], 2)
        combine_out = self.combine(combine_in)
        combine_out = self.reLu(combine_out)

        result = self.out(combine_out)
        result = self.reLu(result)
        return result
