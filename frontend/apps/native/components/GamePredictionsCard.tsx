import { Ionicons } from "@expo/vector-icons";
import { Team, GamePrediction } from "@nba/api/types";
import { useQuery, useMutation } from "@tanstack/react-query";
import React, { useState, useEffect } from "react";
import { View, Text, TouchableOpacity, Alert, Modal, FlatList, SafeAreaView, ActivityIndicator } from "react-native";
import { apiClient } from "@/lib/api-client";

function SelectTeamModal({
  visible,
  onClose,
  onSelectTeam,
  teams,
}: {
  visible: boolean;
  onClose: () => void;
  onSelectTeam: (team: Team) => void;
  teams?: Team[];
}) {
  return (
    <Modal animationType="slide" transparent={false} visible={visible}>
      <SafeAreaView className="flex-1">
        <View className="flex-row justify-between items-center p-4 border-b border-gray-200">
          <Text className="text-lg font-bold">Select a Team</Text>
          <TouchableOpacity onPress={onClose}>
            <Ionicons name="close" size={24} color="black" />
          </TouchableOpacity>
        </View>
        <FlatList
          data={teams}
          keyExtractor={(item) => item.team_id.toString()}
          renderItem={({ item }) => (
            <TouchableOpacity
              className="p-4 border-b border-gray-100"
              onPress={() => onSelectTeam(item)}
            >
              <Text className="text-base">{item.full_name}</Text>
            </TouchableOpacity>
          )}
        />
      </SafeAreaView>
    </Modal>
  );
}

interface GamePredictionsCardProps {
    initialHomeTeam?: Team;
    initialAwayTeam?: Team;
}

export function GamePredictionsCard({ initialHomeTeam, initialAwayTeam }: GamePredictionsCardProps) {
  const [selectedTeams, setSelectedTeams] = useState<{ home?: Team; away?: Team; }>({
      home: initialHomeTeam,
      away: initialAwayTeam,
  });
  const [modalVisible, setModalVisible] = useState(false);
  const [selectingFor, setSelectingFor] = useState<"home" | "away">("home");
  const [predictionResult, setPredictionResult] = useState<GamePrediction | null>(null);

  useEffect(() => {
    setSelectedTeams({ home: initialHomeTeam, away: initialAwayTeam });
    setPredictionResult(null); // Reset prediction when initial teams change
  }, [initialHomeTeam, initialAwayTeam]);

  const { data: teamsData } = useQuery({
    queryKey: ["teams"],
    queryFn: () => apiClient.getTeams(30),
  });

  const predictGameMutation = useMutation({
    mutationFn: ({ homeTeamId, awayTeamId }: { homeTeamId: number; awayTeamId: number; }) =>
      apiClient.predictGame(homeTeamId, awayTeamId),
    onSuccess: (prediction: GamePrediction) => {
      setPredictionResult(prediction);
    },
    onError: () => {
      Alert.alert("Error", "Could not generate prediction. Please try again.");
    },
  });

  const handlePredict = () => {
    if (!selectedTeams.home || !selectedTeams.away) {
      Alert.alert("Error", "Please select both a home and an away team.");
      return;
    }
    if (selectedTeams.home.team_id === selectedTeams.away.team_id) {
      Alert.alert("Error", "Home and away teams cannot be the same.");
      return;
    }
    predictGameMutation.mutate({
      homeTeamId: selectedTeams.home.team_id,
      awayTeamId: selectedTeams.away.team_id,
    });
  };
  
  const handleReset = () => {
      setPredictionResult(null);
      if (!initialHomeTeam && !initialAwayTeam) {
        setSelectedTeams({ home: undefined, away: undefined });
      }
  };

  const openTeamSelection = (type: "home" | "away") => {
    setSelectingFor(type);
    setModalVisible(true);
  };

  const handleSelectTeam = (team: Team) => {
    setSelectedTeams((prev) => ({ ...prev, [selectingFor]: team }));
    setModalVisible(false);
  };

  const renderPrediction = () => {
      if (!predictionResult || !selectedTeams.home || !selectedTeams.away) return null;

      const homeWinProb = (predictionResult.home_win_probability * 100).toFixed(1);
      const awayWinProb = (100 - parseFloat(homeWinProb)).toFixed(1);

      return (
          <View className="space-y-4">
              <View className="items-center">
                  <Text className="text-lg font-bold text-gray-900">Prediction Result</Text>
              </View>
              <View className="flex-row justify-around">
                  <View className="items-center">
                      <Text className="text-3xl font-bold">{homeWinProb}%</Text>
                      <Text className="text-gray-600">{selectedTeams.home.team_abbrev} Win Chance</Text>
                  </View>
                   <View className="items-center">
                      <Text className="text-3xl font-bold">{awayWinProb}%</Text>
                      <Text className="text-gray-600">{selectedTeams.away.team_abbrev} Win Chance</Text>
                  </View>
              </View>
              <View className="items-center p-3 bg-gray-50 rounded-lg">
                  <Text className="text-gray-600 mb-1">Predicted Score</Text>
                  <Text className="text-2xl font-bold tracking-wider">
                      {Math.round(predictionResult.predicted_home_score)} - {Math.round(predictionResult.predicted_away_score)}
                  </Text>
              </View>
              <TouchableOpacity
                className="bg-gray-200 rounded-lg p-3 items-center justify-center"
                onPress={handleReset}
              >
                <Text className="text-gray-800 font-semibold">
                  {initialHomeTeam ? 'Clear Prediction' : 'Predict Another Game'}
                </Text>
              </TouchableOpacity>
          </View>
      );
  }

  const renderForm = () => (
    <View className="space-y-4">
        <View>
          <Text className="text-sm font-medium text-gray-700 mb-2">Home Team</Text>
          <TouchableOpacity
            className="border border-gray-300 rounded-lg p-3"
            onPress={() => openTeamSelection("home")}
            disabled={!!initialHomeTeam}
          >
            <Text className="text-base">{selectedTeams.home?.full_name || "Select Home Team"}</Text>
          </TouchableOpacity>
        </View>

        <View>
          <Text className="text-sm font-medium text-gray-700 mb-2">Away Team</Text>
          <TouchableOpacity
            className="border border-gray-300 rounded-lg p-3"
            onPress={() => openTeamSelection("away")}
            disabled={!!initialAwayTeam}
          >
            <Text className="text-base">{selectedTeams.away?.full_name || "Select Away Team"}</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          className="bg-blue-600 rounded-lg p-3 flex-row items-center justify-center"
          onPress={handlePredict}
          disabled={predictGameMutation.isPending}
        >
          {predictGameMutation.isPending ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text className="text-white font-semibold text-center">Predict Game</Text>
          )}
        </TouchableOpacity>
    </View>
  )

  return (
    <View className="bg-white rounded-lg shadow-sm p-4">
      <SelectTeamModal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        onSelectTeam={handleSelectTeam}
        teams={teamsData?.items}
      />
      <Text className="text-lg font-bold text-gray-900 mb-4">
        Game Predictor
      </Text>

      {predictionResult ? renderPrediction() : renderForm()}
    </View>
  );
}