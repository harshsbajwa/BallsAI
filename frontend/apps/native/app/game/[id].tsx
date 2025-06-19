import { useQuery } from "@tanstack/react-query";
import { useLocalSearchParams } from "expo-router";
import React from "react";
import { View, Text, ScrollView } from "react-native";

import { GamePredictionsCard } from "@/components/GamePredictionsCard";
import { Loading } from "@/components/Loading";

import { apiClient } from "@/lib/api-client";

export default function GameDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const gameId = parseInt(id, 10);

  const { data: game, isLoading } = useQuery({
    queryKey: ["game", gameId],
    queryFn: () => apiClient.getGame(gameId),
    enabled: !!gameId,
  });

  if (isLoading) {
    return (
      <View className="flex-1 bg-gray-50">
        <Loading text="Loading game..." />
      </View>
    );
  }

  if (!game) {
    return (
      <View className="flex-1 bg-gray-50 items-center justify-center">
        <Text className="text-red-500 text-lg">Game not found</Text>
      </View>
    );
  }

  const isUpcoming = game.home_score === null || game.away_score === null;
  const gameTime = new Date(game.game_date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <ScrollView className="flex-1 bg-gray-50 p-4">
      <View className="bg-white rounded-lg shadow-sm p-6 mb-4">
        {/* Teams */}
        <View className="flex-row justify-between items-center">
          <View className="flex-1 items-center">
            <Text className="text-xl font-bold">{game.away_team.full_name}</Text>
            <Text className="text-lg text-gray-500">Away</Text>
          </View>
          <Text className="text-2xl font-bold text-gray-400 mx-4">VS</Text>
          <View className="flex-1 items-center">
            <Text className="text-xl font-bold">{game.home_team.full_name}</Text>
            <Text className="text-lg text-gray-500">Home</Text>
          </View>
        </View>

        {/* Score / Time */}
        <View className="items-center mt-6">
          {isUpcoming ? (
            <Text className="text-3xl font-bold text-gray-800">{gameTime}</Text>
          ) : (
            <View className="flex-row items-center">
              <Text className="text-5xl font-bold">{game.away_score}</Text>
              <Text className="text-5xl font-bold mx-4">-</Text>
              <Text className="text-5xl font-bold">{game.home_score}</Text>
            </View>
          )}
          <Text className="text-md text-gray-400 mt-2">
            {new Date(game.game_date).toLocaleDateString()}
          </Text>
        </View>
      </View>

      {/* Prediction */}
      <GamePredictionsCard
        initialHomeTeam={game.home_team}
        initialAwayTeam={game.away_team}
      />
    </ScrollView>
  );
}