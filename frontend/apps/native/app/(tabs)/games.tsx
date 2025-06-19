import React from "react";
import { View, Text, FlatList, RefreshControl, TouchableOpacity } from "react-native";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Loading } from "@/components/Loading";
import { Game } from "@nba/api/types";
import { Link } from "expo-router";

export default function GamesScreen() {
  const {
    data: todaysGames,
    isLoading,
    isRefetching,
    error,
    refetch,
  } = useQuery({
    queryKey: ["games", "today"],
    queryFn: () => apiClient.getTodaysGames(),
  });

  const renderGame = ({ item }: { item: Game }) => (
    <Link href={`/game/${item.game_id}`} asChild>
      <TouchableOpacity className="bg-white mx-4 my-2 rounded-lg p-4 shadow-sm active:opacity-70">
        <View className="flex-row justify-between items-center">
          <View className="flex-1">
            <View className="flex-row items-center justify-between mb-2">
              <Text className="font-semibold text-lg">
                {item.away_team.team_abbrev}
              </Text>
              <Text className="text-gray-500">@</Text>
              <Text className="font-semibold text-lg">
                {item.home_team.team_abbrev}
              </Text>
            </View>
            <View className="flex-row items-center justify-between">
              <Text className="text-gray-600">{item.away_team.full_name}</Text>
              <Text className="text-gray-600">{item.home_team.full_name}</Text>
            </View>
          </View>
          <View className="ml-4 items-end w-16">
            {item.away_score !== null && item.home_score !== null ? (
              <View className="items-center">
                <Text className="font-bold text-xl text-blue-600">
                  {item.away_score}
                </Text>
                <Text className="font-bold text-xl text-blue-600">
                  {item.home_score}
                </Text>
              </View>
            ) : (
              <Text className="text-gray-500 text-sm">
                {new Date(item.game_date).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </Text>
            )}
          </View>
        </View>
        <Text className="text-gray-400 text-xs mt-2">
          {new Date(item.game_date).toLocaleDateString()}
        </Text>
      </TouchableOpacity>
    </Link>
  );

  if (isLoading && !isRefetching) {
    return (
      <View className="flex-1 bg-gray-50">
        <Loading text="Loading games..." />
      </View>
    );
  }

  if (error) {
    return (
      <View className="flex-1 bg-gray-50 items-center justify-center">
        <Text className="text-red-500 text-lg">Error loading games</Text>
        <Text className="text-gray-500 text-sm mt-2">Please try again</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-gray-50 pt-4">
      {todaysGames?.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <Text className="text-gray-500 text-lg">No games scheduled today</Text>
          <Text className="text-gray-400 text-sm mt-2">
            Check back later for updates
          </Text>
        </View>
      ) : (
        <FlatList
          data={todaysGames}
          keyExtractor={(item) => item.game_id.toString()}
          renderItem={renderGame}
          refreshControl={
            <RefreshControl refreshing={isRefetching} onRefresh={refetch} />
          }
          contentInsetAdjustmentBehavior="automatic"
          showsVerticalScrollIndicator={false}
        />
      )}
    </View>
  );
}